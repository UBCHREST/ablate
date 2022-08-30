#include "boundarySolver.hpp"
#include <set>
#include <utility>
#include "boundaryProcess.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/stringUtilities.hpp"

ablate::boundarySolver::BoundarySolver::BoundarySolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> fieldBoundary,
                                                       std::vector<std::shared_ptr<BoundaryProcess>> boundaryProcesses, std::shared_ptr<parameters::Parameters> options, bool mergeFaces)
    : CellSolver(std::move(solverId), std::move(region), std::move(options)), fieldBoundary(std::move(fieldBoundary)), boundaryProcesses(std::move(boundaryProcesses)), mergeFaces(mergeFaces) {}

ablate::boundarySolver::BoundarySolver::~BoundarySolver() {
    if (gradientCalculator) {
        PetscFVDestroy(&gradientCalculator);
    }
}

static void AddNeighborsToStencil(const std::shared_ptr<ablate::domain::SubDomain>& subdomain, std::set<PetscInt>& stencilSet, DMLabel boundaryLabel, PetscInt boundaryValue, PetscInt depth, DM dm,
                                  PetscInt cell, PetscInt maxDepth) {
    // Check to see if this cell is already in the list
    if (stencilSet.count(cell)) {
        return;
    }

    // do not allow this stencil to go to another subdomain
    if (!subdomain->InRegion(cell)) {
        return;
    }

    // Add to the list
    stencilSet.insert(cell);

    // If not at max depth, check another layer
    if (depth < maxDepth) {
        PetscInt numberFaces;
        const PetscInt* cellFaces;
        DMPlexGetConeSize(dm, cell, &numberFaces) >> ablate::checkError;
        DMPlexGetCone(dm, cell, &cellFaces) >> ablate::checkError;

        // For each connected face
        for (PetscInt f = 0; f < numberFaces; f++) {
            PetscInt face = cellFaces[f];

            // Don't allow the search back over the boundary Label
            PetscInt faceValue;
            DMLabelGetValue(boundaryLabel, face, &faceValue) >> ablate::checkError;
            if (faceValue == boundaryValue) {
                continue;
            }

            // check any neighbors
            PetscInt numberNeighborCells;
            const PetscInt* neighborCells;
            DMPlexGetSupportSize(dm, face, &numberNeighborCells) >> ablate::checkError;
            DMPlexGetSupport(dm, face, &neighborCells) >> ablate::checkError;
            for (PetscInt n = 0; n < numberNeighborCells; n++) {
                AddNeighborsToStencil(subdomain, stencilSet, boundaryLabel, boundaryValue, depth + 1, dm, neighborCells[n], maxDepth);
            }
        }
    }
}

void ablate::boundarySolver::BoundarySolver::Setup() {
    ablate::solver::CellSolver::Setup();

    // do a simple sanity check for labels
    GetRegion()->CheckForLabel(subDomain->GetDM());
    fieldBoundary->CheckForLabel(subDomain->GetDM());

    // march over process and link to the flow
    for (auto& process : boundaryProcesses) {
        process->Initialize(*this);
    }

    // Set up the gradient calculator
    PetscFVCreate(PETSC_COMM_SELF, &gradientCalculator) >> checkError;
    // Set least squares as the default type
    PetscFVSetType(gradientCalculator, PETSCFVLEASTSQUARES) >> checkError;
    // Set any other required options
    PetscObjectSetOptions((PetscObject)gradientCalculator, petscOptions) >> checkError;
    PetscFVSetFromOptions(gradientCalculator) >> checkError;
    PetscFVSetNumComponents(gradientCalculator, 1) >> checkError;
    PetscFVSetSpatialDimension(gradientCalculator, subDomain->GetDimensions()) >> checkError;

    // Make sure that his fvm supports gradients
    PetscBool supportsGrads;
    PetscFVGetComputeGradients(gradientCalculator, &supportsGrads) >> checkError;
    if (!supportsGrads) {
        throw std::invalid_argument("The BoundarySolver requires a PetscFVM that supports gradients.");
    }

    // Get the geometry for the mesh
    DM faceDM, cellDM;
    VecGetDM(faceGeomVec, &faceDM) >> checkError;
    VecGetDM(cellGeomVec, &cellDM) >> checkError;
    const PetscScalar* cellGeomArray;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
    PetscInt dim = subDomain->GetDimensions();

    // Get the labels
    DMLabel boundaryLabel;
    PetscInt boundaryValue = fieldBoundary->GetValue();
    DMGetLabel(subDomain->GetDM(), fieldBoundary->GetName().c_str(), &boundaryLabel) >> checkError;

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;

    // compute the max depth
    PetscInt maxCellDepth = dim;

    // March over each cell in this region to create the stencil
    solver::Range cellRange;
    GetCellRange(cellRange);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // make sure we are not working on a ghost cell
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, cell, &ghost);
        }
        if (ghost >= 0) {
            continue;
        }

        // If merging faces, march over the multiple faces per cell
        if (mergeFaces) {
            // keep a list of cells in the stencil
            std::set<PetscInt> stencilSet{cell};

            // March over each face
            PetscInt numberFaces;
            const PetscInt* cellFaces;
            DMPlexGetConeSize(subDomain->GetDM(), cell, &numberFaces) >> checkError;
            DMPlexGetCone(subDomain->GetDM(), cell, &cellFaces) >> checkError;

            // Create a new BoundaryFVFaceGeom
            BoundaryFVFaceGeom geom{.normal = {0.0, 0.0, 0.0}, .areas = {0.0, 0.0, 0.0}, .centroid = {0.0, 0.0, 0.0}};

            // Perform some error checking
            if (numberFaces < 1) {
                throw std::runtime_error("Isolated cell " + std::to_string(cell) + " cannot be used in BoundarySolver.");
            }

            // set the faceId and increment
            geom.faceId = cellFaces[0];

            // For each connected face
            PetscInt usedFaceCount = 0;
            PetscInt referenceCell = -1;  // The first cell connected to this face
            for (PetscInt f = 0; f < numberFaces; f++) {
                PetscInt face = cellFaces[f];

                // check to see if this face is in the boundary region
                PetscInt faceValue;
                DMLabelGetValue(boundaryLabel, face, &faceValue) >> checkError;
                if (faceValue != boundaryValue) {
                    continue;
                }

                // Get the connected cells
                PetscInt numberNeighborCells;
                const PetscInt* neighborCells;
                DMPlexGetSupportSize(subDomain->GetDM(), face, &numberNeighborCells) >> checkError;
                DMPlexGetSupport(subDomain->GetDM(), face, &neighborCells) >> checkError;
                referenceCell = neighborCells[0] == cell ? neighborCells[1] : neighborCells[0];

                for (PetscInt n = 0; n < numberNeighborCells; n++) {
                    AddNeighborsToStencil(subDomain, stencilSet, boundaryLabel, boundaryValue, 1, subDomain->GetDM(), neighborCells[n], maxCellDepth);
                }

                // Add this geometry to the BoundaryFVFaceGeom
                PetscFVFaceGeom* fg;
                DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;

                // The normal should be pointing away from the other phase into the boundary solver domain.  The current fg support points from cell[0] -> cell[1]
                // If the neighborCells[0] is in the boundary (this cell), flip the normal
                if (neighborCells[0] == cell) {
                    for (PetscInt d = 0; d < dim; d++) {
                        geom.normal[d] -= fg->normal[d];
                        geom.areas[d] -= fg->normal[d];
                        geom.centroid[d] += fg->centroid[d];
                    }
                } else {
                    for (PetscInt d = 0; d < dim; d++) {
                        geom.normal[d] += fg->normal[d];
                        geom.areas[d] += fg->normal[d];
                        geom.centroid[d] += fg->centroid[d];
                    }
                }
                usedFaceCount++;
            }

            // take average face location
            for (PetscInt d = 0; d < dim; d++) {
                geom.centroid[d] /= usedFaceCount;
            }

            // compute the normal
            utilities::MathUtilities::NormVector(dim, geom.normal);

            // remove the boundary cell from the stencil
            stencilSet.erase(cell);
            stencilSet.erase(referenceCell);

            // Add the stencil gradient
            std::vector<PetscInt> stencil{referenceCell};
            stencil.insert(stencil.end(), stencilSet.begin(), stencilSet.end());
            CreateGradientStencil(cell, geom, stencil, cellDM, cellGeomArray);
        } else {
            // March over each face
            PetscInt numberFaces;
            const PetscInt* cellFaces;
            DMPlexGetConeSize(subDomain->GetDM(), cell, &numberFaces) >> checkError;
            DMPlexGetCone(subDomain->GetDM(), cell, &cellFaces) >> checkError;

            // Perform some error checking
            if (numberFaces < 1) {
                throw std::runtime_error("Isolated cell " + std::to_string(cell) + " cannot be used in BoundarySolver.");
            }

            // Create a different stencil per face
            for (PetscInt f = 0; f < numberFaces; f++) {
                // keep a list of cells in the stencil
                std::set<PetscInt> stencilSet{cell};

                // extract the face
                PetscInt face = cellFaces[f];

                // check to see if this face is in the boundary region
                PetscInt faceValue;
                DMLabelGetValue(boundaryLabel, face, &faceValue) >> checkError;
                if (faceValue != boundaryValue) {
                    continue;
                }

                // Create a new BoundaryFVFaceGeom
                BoundaryFVFaceGeom geom{.faceId = face, .normal = {0.0, 0.0, 0.0}, .areas = {0.0, 0.0, 0.0}, .centroid = {0.0, 0.0, 0.0}};

                // Get the connected cells
                PetscInt numberNeighborCells;
                const PetscInt* neighborCells;
                DMPlexGetSupportSize(subDomain->GetDM(), face, &numberNeighborCells) >> checkError;
                DMPlexGetSupport(subDomain->GetDM(), face, &neighborCells) >> checkError;

                for (PetscInt n = 0; n < numberNeighborCells; n++) {
                    AddNeighborsToStencil(subDomain, stencilSet, boundaryLabel, boundaryValue, 1, subDomain->GetDM(), neighborCells[n], maxCellDepth);
                }

                // Add this geometry to the BoundaryFVFaceGeom
                PetscFVFaceGeom* fg;
                DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;

                // The normal should be pointing away from the other phase into the boundary solver domain.  The current fg support points from cell[0] -> cell[1]
                PetscInt referenceCell = neighborCells[0] == cell ? neighborCells[1] : neighborCells[0];
                // If the neighborCells[0] is in the boundary (this cell), flip the normal
                if (neighborCells[0] == cell) {
                    for (PetscInt d = 0; d < dim; d++) {
                        geom.normal[d] -= fg->normal[d];
                        geom.areas[d] -= fg->normal[d];
                        geom.centroid[d] += fg->centroid[d];
                    }
                } else {
                    for (PetscInt d = 0; d < dim; d++) {
                        geom.normal[d] += fg->normal[d];
                        geom.areas[d] += fg->normal[d];
                        geom.centroid[d] += fg->centroid[d];
                    }
                }

                // compute the normal
                utilities::MathUtilities::NormVector(dim, geom.normal);

                // remove the boundary cell from the stencil
                stencilSet.erase(cell);
                stencilSet.erase(referenceCell);

                // Add the stencil gradient
                std::vector<PetscInt> stencil{referenceCell};
                stencil.insert(stencil.end(), stencilSet.begin(), stencilSet.end());
                CreateGradientStencil(cell, geom, stencil, cellDM, cellGeomArray);
            }
        }
    }
    RestoreRange(cellRange);

    // clean up the geom
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
}
void ablate::boundarySolver::BoundarySolver::Initialize() {
    RegisterPreStage([](auto ts, auto& solver, auto stageTime) {
        Vec locXVec;
        DMGetLocalVector(solver.GetSubDomain().GetDM(), &locXVec) >> checkError;
        DMGlobalToLocal(solver.GetSubDomain().GetDM(), solver.GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> checkError;

        // Get the time from the ts
        PetscReal time;
        TSGetTime(ts, &time) >> checkError;

        auto& cellSolver = dynamic_cast<CellSolver&>(solver);
        cellSolver.UpdateAuxFields(time, locXVec, solver.GetSubDomain().GetAuxVector());

        DMRestoreLocalVector(solver.GetSubDomain().GetDM(), &locXVec) >> checkError;
    });

    if (!boundaryUpdateFunctions.empty()) {
        RegisterPreStep([this](auto ts, auto& solver) { UpdateVariablesPreStep(ts, solver); });
    }
}
void ablate::boundarySolver::BoundarySolver::RegisterFunction(ablate::boundarySolver::BoundarySolver::BoundarySourceFunction function, void* context, const std::vector<std::string>& sourceFields,
                                                              const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields, BoundarySourceType type) {
    // Create the FVMRHS Function
    BoundarySourceFunctionDescription functionDescription{.function = function, .context = context, .type = type};

    for (auto& inputField : inputFields) {
        auto& inputFieldId = subDomain->GetField(inputField);
        functionDescription.inputFieldsOffset.push_back(inputFieldId.offset);
    }

    for (const auto& auxField : auxFields) {
        auto& auxFieldId = subDomain->GetField(auxField);
        functionDescription.auxFieldsOffset.push_back(auxFieldId.offset);
    }

    if (type == BoundarySourceType::Face) {
        // add to the outputComponents list for output later
        for (const auto& sourceField : sourceFields) {
            auto componentLoc = find(outputComponents.begin(), outputComponents.end(), sourceField);

            // If this is the end (not found) add to the list
            if (componentLoc == outputComponents.end()) {
                functionDescription.sourceFieldsOffset.push_back((PetscInt)outputComponents.size());
                outputComponents.push_back(sourceField);
            } else {
                // it was found, just add component
                functionDescription.sourceFieldsOffset.push_back((PetscInt)std::distance(componentLoc, outputComponents.begin()));
            }
        }

        boundaryOutputFunctions.push_back(functionDescription);
    } else {
        // check the subdomain for information about the fields
        for (auto& sourceField : sourceFields) {
            auto& fieldId = subDomain->GetField(sourceField);
            functionDescription.sourceFieldsOffset.push_back(fieldId.offset);
        }

        boundarySourceFunctions.push_back(functionDescription);
    }
}

void ablate::boundarySolver::BoundarySolver::RegisterFunction(ablate::boundarySolver::BoundarySolver::BoundaryUpdateFunction function, void* context, const std::vector<std::string>& inputFields,
                                                              const std::vector<std::string>& auxFields) {
    BoundaryUpdateFunctionDescription functionDescription{.function = function, .context = context};

    for (auto& inputField : inputFields) {
        auto& inputFieldId = subDomain->GetField(inputField);
        functionDescription.inputFields.push_back(inputFieldId.subId);
    }

    for (const auto& auxField : auxFields) {
        auto& auxFieldId = subDomain->GetField(auxField);
        functionDescription.auxFields.push_back(auxFieldId.subId);
    }

    boundaryUpdateFunctions.push_back(functionDescription);
}
PetscErrorCode ablate::boundarySolver::BoundarySolver::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) {
    PetscFunctionBeginUser;
    PetscCall(ComputeRHSFunction(time, locXVec, locFVec, boundarySourceFunctions));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::boundarySolver::BoundarySolver::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec,
                                                                          const std::vector<BoundarySourceFunctionDescription>& activeBoundarySourceFunctions) {
    PetscFunctionBeginUser;

    // Extract the cell geometry, and the dm that holds the information
    auto dm = subDomain->GetDM();
    auto auxDM = subDomain->GetAuxDM();
    auto dim = subDomain->GetDimensions();
    DM dmCell;
    const PetscScalar* cellGeomArray;
    PetscCall(VecGetDM(cellGeomVec, &dmCell));
    PetscCall(VecGetArrayRead(cellGeomVec, &cellGeomArray));

    // prepare to compute the source, u, and a offsets
    PetscInt nf;
    PetscCall(PetscDSGetNumFields(subDomain->GetDiscreteSystem(), &nf));

    // Create the required offset arrays. These are sized for the max possible value
    PetscInt* offsetsTotal;
    PetscCall(PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &offsetsTotal));
    PetscInt* auxOffTotal = nullptr;
    if (auto auxDS = subDomain->GetAuxDiscreteSystem()) {
        PetscCall(PetscDSGetComponentOffsets(auxDS, &auxOffTotal));
    }

    // Get the size of the field
    PetscInt scratchSize;
    PetscCall(PetscDSGetTotalDimension(subDomain->GetDiscreteSystem(), &scratchSize));
    std::vector<PetscScalar> distributedSourceScratch(scratchSize);

    // Get the region to march over
    if (!gradientStencils.empty()) {
        // Get pointers to sol, aux, and f vectors
        const PetscScalar *locXArray, *locAuxArray = nullptr;
        PetscScalar* locFArray;
        PetscCall(VecGetArrayRead(locXVec, &locXArray));
        if (auto locAuxVec = subDomain->GetAuxVector()) {
            PetscCall(VecGetArrayRead(locAuxVec, &locAuxArray));
        }
        PetscCall(VecGetArray(locFVec, &locFArray));

        // Store pointers to the stencil variables
        std::vector<const PetscScalar*> inputStencilValues(maximumStencilSize);
        std::vector<const PetscScalar*> auxStencilValues(maximumStencilSize);

        // March over each boundary function
        for (const auto& function : activeBoundarySourceFunctions) {

            auto sourceOffsetsPointer = function.sourceFieldsOffset.data();
            auto inputOffsetsPointer = function.inputFieldsOffset.data();
            auto auxOffsetsPointer = function.auxFieldsOffset.data();

            // March over each cell in this region
            for (const auto& stencilInfo : gradientStencils) {
                // Get the cell geom
                const PetscFVCellGeom* cg;
                PetscCall(DMPlexPointLocalRead(dmCell, stencilInfo.cellId, cellGeomArray, &cg));

                // Get pointers to the area of interest
                const PetscScalar *solPt, *auxPt = nullptr;
                PetscCall(DMPlexPointLocalRead(dm, stencilInfo.cellId, locXArray, &solPt));
                if (auxDM) {
                    PetscCall(DMPlexPointLocalRead(auxDM, stencilInfo.cellId, locAuxArray, &auxPt));
                }

                // Get each of the stencil pts
                for (PetscInt p = 0; p < stencilInfo.stencilSize; p++) {
                    PetscCall(DMPlexPointLocalRead(dm, stencilInfo.stencil[p], locXArray, &inputStencilValues[p]));
                    if (auxDM) {
                        PetscCall(DMPlexPointLocalRead(auxDM, stencilInfo.stencil[p], locAuxArray, &auxStencilValues[p]));
                    }
                }

                // Get the pointer to the rhs
                switch (function.type) {
                    case BoundarySourceType::Point:
                        PetscScalar* rhs;
                        PetscCall(DMPlexPointLocalRef(dm, stencilInfo.cellId, locFArray, &rhs));

                        /*PetscErrorCode (*)(PetscInt dim, const BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell,
                                           const PetscInt uOff[], const PetscScalar* boundaryValues, const PetscScalar* stencilValues[],
                                           const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                           PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[], void* ctx)*/
                        PetscCall(function.function(dim,
                                                    &stencilInfo.geometry,
                                                    cg,
                                                    inputOffsetsPointer,
                                                    solPt,
                                                    inputStencilValues.data(),
                                                    auxOffsetsPointer,
                                                    auxPt,
                                                    auxStencilValues.data(),
                                                    stencilInfo.stencilSize,
                                                    stencilInfo.stencil.data(),
                                                    stencilInfo.gradientWeights.data(),
                                                    sourceOffsetsPointer,
                                                    rhs,
                                                    function.context));
                        break;
                    case BoundarySourceType::Distributed:
                        /*PetscErrorCode (*)(PetscInt dim, const BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell,
                                                                   const PetscInt uOff[], const PetscScalar* boundaryValues, const PetscScalar* stencilValues[],
                                                                   const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                                                   PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[],
                           void* ctx)*/

                        // zero out the distributedSourceScratch
                        PetscCall(PetscArrayzero(distributedSourceScratch.data(), (PetscInt)distributedSourceScratch.size()));

                        PetscCall(function.function(dim,
                                                    &stencilInfo.geometry,
                                                    cg,
                                                    inputOffsetsPointer,
                                                    solPt,
                                                    inputStencilValues.data(),
                                                    auxOffsetsPointer,
                                                    auxPt,
                                                    auxStencilValues.data(),
                                                    stencilInfo.stencilSize,
                                                    stencilInfo.stencil.data(),
                                                    stencilInfo.gradientWeights.data(),
                                                    sourceOffsetsPointer,
                                                    distributedSourceScratch.data(),
                                                    function.context));

                        // Now distribute to each stencil point
                        for (PetscInt s = 0; s < stencilInfo.stencilSize; ++s) {
                            // Get the point in the rhs for this point.  It might be ghost but that is ok, the values are added together later
                            PetscCall(DMPlexPointLocalRef(dm, stencilInfo.stencil[s], locFArray, &rhs));

                            // Now over the entire rhs, the function should have added the values correctly using the sourceOffsetsPointer
                            for (PetscInt sc = 0; sc < scratchSize; sc++) {
                                rhs[sc] += (distributedSourceScratch[sc] * stencilInfo.distributionWeights[s]) / stencilInfo.volumes[s];
                            }
                        }

                        break;
                    case BoundarySourceType::Flux:
                        // zero out the distributedSourceScratch
                        PetscCall(PetscArrayzero(distributedSourceScratch.data(), (PetscInt)distributedSourceScratch.size()));

                        PetscCall(function.function(dim,
                                                    &stencilInfo.geometry,
                                                    cg,
                                                    inputOffsetsPointer,
                                                    solPt,
                                                    inputStencilValues.data(),
                                                    auxOffsetsPointer,
                                                    auxPt,
                                                    auxStencilValues.data(),
                                                    stencilInfo.stencilSize,
                                                    stencilInfo.stencil.data(),
                                                    stencilInfo.gradientWeights.data(),
                                                    sourceOffsetsPointer,
                                                    distributedSourceScratch.data(),
                                                    function.context));

                        // the first cell in the stencil is always the neighbor cell
                        // Get the point in the rhs for this point.  It might be ghost but that is ok, the values are added together later
                        PetscCall(DMPlexPointLocalRef(dm, stencilInfo.stencil.front(), locFArray, &rhs));

                        // Now over the entire rhs, the function should have added the values correctly using the sourceOffsetsPointer
                        for (PetscInt sc = 0; sc < scratchSize; sc++) {
                            rhs[sc] += distributedSourceScratch[sc] / stencilInfo.volumes.front();
                        }

                        break;

                    case BoundarySourceType::Face:
                        // Get the vec DM from the locFArray
                        DM vecDm;
                        PetscCall(VecGetDM(locFVec, &vecDm));

                        // Assume that the right hand side vector is for face information
                        PetscScalar* faceRhs;
                        PetscCall(DMPlexPointLocalRef(vecDm, stencilInfo.geometry.faceId, locFArray, &faceRhs));

                        /*PetscErrorCode (*)(PetscInt dim, const BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell,
                                           const PetscInt uOff[], const PetscScalar* boundaryValues, const PetscScalar* stencilValues[],
                                           const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                           PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[], void* ctx)*/
                        PetscCall(function.function(dim,
                                                    &stencilInfo.geometry,
                                                    cg,
                                                    inputOffsetsPointer,
                                                    solPt,
                                                    inputStencilValues.data(),
                                                    auxOffsetsPointer,
                                                    auxPt,
                                                    auxStencilValues.data(),
                                                    stencilInfo.stencilSize,
                                                    stencilInfo.stencil.data(),
                                                    stencilInfo.gradientWeights.data(),
                                                    sourceOffsetsPointer,
                                                    faceRhs,
                                                    function.context));
                        break;
                }
            }
        }

        // clean up access
        PetscCall(VecRestoreArrayRead(locXVec, &locXArray));
        if (auto locAuxVec = subDomain->GetAuxVector()) {
            PetscCall(VecRestoreArrayRead(locAuxVec, &locAuxArray));
        }
        PetscCall(VecRestoreArray(locFVec, &locFArray));

        // clean up the geom
        PetscCall(VecRestoreArrayRead(cellGeomVec, &cellGeomArray));
    }

    PetscFunctionReturn(0);
}
void ablate::boundarySolver::BoundarySolver::InsertFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, PetscReal time) {
    for (const auto& fieldFunction : fieldFunctions) {
        // Get the field
        const auto& field = subDomain->GetField(fieldFunction->GetName());

        // Get the vec that goes with the field
        auto vec = subDomain->GetVec(field);
        auto dm = subDomain->GetFieldDM(field);

        // Get the raw array
        PetscScalar* array;
        VecGetArray(vec, &array) >> checkError;

        // March over each cell
        solver::Range cellRange;
        GetCellRange(cellRange);
        PetscInt dim = subDomain->GetDimensions();

        // Get the petscFunction/context
        auto petscFunction = fieldFunction->GetFieldFunction()->GetPetscFunction();
        auto context = fieldFunction->GetFieldFunction()->GetContext();

        // March over each cell in this region
        PetscInt cOffset = 0;  // Keep track of the cell offset
        for (PetscInt c = cellRange.start; c < cellRange.end; ++c, cOffset++) {
            // if there is a cell array, use it, otherwise it is just c
            const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

            // Get pointers to sol, aux, and f vectors
            PetscScalar* pt = nullptr;
            switch (field.location) {
                case domain::FieldLocation::SOL:
                    DMPlexPointGlobalFieldRef(dm, cell, field.id, array, &pt) >> checkError;
                    break;
                case domain::FieldLocation::AUX:
                    DMPlexPointLocalFieldRef(dm, cell, field.id, array, &pt) >> checkError;
                    break;
            }
            if (pt) {
                petscFunction(dim, time, gradientStencils[cOffset].geometry.centroid, field.numberComponents, pt, context);
            }
        }

        RestoreRange(cellRange);
        VecRestoreArray(vec, &array);
    }
}

void ablate::boundarySolver::BoundarySolver::ComputeGradient(PetscInt dim, PetscScalar boundaryValue, PetscInt stencilSize, const PetscScalar* stencilValues, const PetscScalar* stencilWeights,
                                                             PetscScalar* grad) {
    PetscArrayzero(grad, dim);

    for (PetscInt c = 0; c < stencilSize; ++c) {
        PetscScalar delta = stencilValues[c] - boundaryValue;

        for (PetscInt d = 0; d < dim; ++d) {
            grad[d] += stencilWeights[c * dim + d] * delta;
        }
    }
}

void ablate::boundarySolver::BoundarySolver::ComputeGradientAlongNormal(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom* fg, PetscScalar boundaryValue,
                                                                        PetscInt stencilSize, const PetscScalar* stencilValues, const PetscScalar* stencilWeights, PetscScalar& dPhiDNorm) {
    dPhiDNorm = 0.0;
    for (PetscInt c = 0; c < stencilSize; ++c) {
        PetscScalar delta = stencilValues[c] - boundaryValue;

        for (PetscInt d = 0; d < dim; ++d) {
            dPhiDNorm += stencilWeights[c * dim + d] * delta * fg->normal[d];
        }
    }
}

std::vector<ablate::boundarySolver::BoundarySolver::GradientStencil> ablate::boundarySolver::BoundarySolver::GetBoundaryGeometry(PetscInt cell) const {
    std::vector<ablate::boundarySolver::BoundarySolver::GradientStencil> searchResult;
    std::copy_if(gradientStencils.begin(), gradientStencils.end(), std::back_inserter(searchResult), [cell](const auto& stencil) { return stencil.cellId == cell; });
    return searchResult;
}

void ablate::boundarySolver::BoundarySolver::CreateGradientStencil(PetscInt cellId, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom& geometry, const std::vector<PetscInt>& stencil,
                                                                   DM cellDM, const PetscScalar* cellGeomArray) {
    // Compute the weights for the stencil
    auto newStencil = GradientStencil{.cellId = cellId, .geometry = geometry, .stencil = stencil, .stencilSize = (PetscInt)stencil.size()};

    // resize stencil weights
    auto dim = subDomain->GetDimensions();
    newStencil.gradientWeights.resize(newStencil.stencilSize * dim, 0.0);
    // Use a Reciprocal distance interpolate for the distribution weights.  This can be abstracted away in the future.
    newStencil.distributionWeights.resize(newStencil.stencilSize * dim, 0.0);
    newStencil.volumes.resize(newStencil.stencilSize, 0.0);

    // Size up the dx for scratch space
    std::vector<PetscScalar> dx(newStencil.stencilSize * dim);

    // add in each cell contribution
    PetscScalar distributionWeightSum = 0.0;
    for (std::size_t n = 0; n < stencil.size(); n++) {
        PetscFVCellGeom* cg;
        DMPlexPointLocalRead(cellDM, stencil[n], cellGeomArray, &cg);
        for (PetscInt d = 0; d < dim; ++d) {
            dx[n * dim + d] = cg->centroid[d] - newStencil.geometry.centroid[d];
            newStencil.distributionWeights[n] += PetscSqr(cg->centroid[d] - newStencil.geometry.centroid[d]);
        }
        newStencil.distributionWeights[n] = 1.0 / PetscSqrtScalar(newStencil.distributionWeights[n]);
        distributionWeightSum += newStencil.distributionWeights[n];

        // store the volume
        newStencil.volumes[n] = cg->volume;
    }

    // normalize the distributionWeights
    utilities::MathUtilities::ScaleVector(newStencil.distributionWeights.size(), newStencil.distributionWeights.data(), 1.0 / distributionWeightSum);

    // Reset the least squares calculator if needed
    if ((PetscInt)stencil.size() > maximumStencilSize) {
        maximumStencilSize = (PetscInt)stencil.size();
        PetscFVLeastSquaresSetMaxFaces(gradientCalculator, maximumStencilSize) >> checkError;
    }
    PetscFVComputeGradient(gradientCalculator, (PetscInt)stencil.size(), dx.data(), newStencil.gradientWeights.data()) >> checkError;

    // Store the stencil
    gradientStencils.push_back(std::move(newStencil));
}
void ablate::boundarySolver::BoundarySolver::UpdateVariablesPreStep(TS, ablate::solver::Solver&) {
    // Extract the cell geometry, and the dm that holds the information
    auto dm = subDomain->GetDM();
    auto auxDM = subDomain->GetAuxDM();
    auto dim = subDomain->GetDimensions();
    DM dmCell;
    const PetscScalar* cellGeomArray;
    VecGetDM(cellGeomVec, &dmCell) >> checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    // get the local x vector for ghost node information
    Vec locXVec;
    DMGetLocalVector(subDomain->GetDM(), &locXVec) >> checkError;
    DMGlobalToLocalBegin(subDomain->GetDM(), subDomain->GetSolutionVector(), INSERT_VALUES, locXVec) >> checkError;

    // prepare to compute the source, u, and a offsets
    PetscInt nf;
    PetscDSGetNumFields(subDomain->GetDiscreteSystem(), &nf) >> checkError;

    // Create the required offset arrays. These are sized for the max possible value
    PetscInt* offsetsTotal;
    PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &offsetsTotal) >> checkError;
    PetscInt* auxOffTotal = nullptr;
    if (auto auxDS = subDomain->GetAuxDiscreteSystem()) {
        PetscDSGetComponentOffsets(auxDS, &auxOffTotal) >> checkError;
    }

    // Get the size of the field
    PetscInt scratchSize;
    PetscDSGetTotalDimension(subDomain->GetDiscreteSystem(), &scratchSize) >> checkError;
    std::vector<PetscScalar> distributedSourceScratch(scratchSize);

    // presize the offsets
    std::vector<PetscInt> sourceOffsets(subDomain->GetFields().size(), -1);
    std::vector<PetscInt> inputOffsets(subDomain->GetFields().size(), -1);
    std::vector<PetscInt> auxOffsets(subDomain->GetFields(domain::FieldLocation::AUX).size(), -1);

    DMGlobalToLocalEnd(subDomain->GetDM(), subDomain->GetSolutionVector(), INSERT_VALUES, locXVec) >> checkError;

    // Get the region to march over
    if (!gradientStencils.empty()) {
        // Get pointers to sol, aux, and f vectors
        PetscScalar *globXArray, *locAuxArray = nullptr;
        VecGetArray(subDomain->GetSolutionVector(), &globXArray);
        if (auto locAuxVec = subDomain->GetAuxVector()) {
            VecGetArray(locAuxVec, &locAuxArray) >> checkError;
        }

        const PetscScalar* localXArray;
        VecGetArrayRead(locXVec, &localXArray);

        // March over each boundary function
        for (const auto& function : boundaryUpdateFunctions) {
            for (std::size_t i = 0; i < function.inputFields.size(); i++) {
                inputOffsets[i] = offsetsTotal[function.inputFields[i]];
            }
            for (std::size_t i = 0; i < function.auxFields.size(); i++) {
                auxOffsets[i] = auxOffTotal[function.auxFields[i]];
            }

            auto inputOffsetsPointer = inputOffsets.data();
            auto auxOffsetsPointer = auxOffsets.data();

            // March over each cell in this region
            for (const auto& stencilInfo : gradientStencils) {
                if (!stencilInfo.stencilSize) {
                    continue;
                }

                // Get the cell geom
                const PetscFVCellGeom* cg;
                DMPlexPointLocalRead(dmCell, stencilInfo.cellId, cellGeomArray, &cg) >> checkError;

                // Get pointers to the area of interest
                PetscScalar *solPt = nullptr, *auxPt = nullptr;
                DMPlexPointGlobalRef(dm, stencilInfo.cellId, globXArray, &solPt) >> checkError;
                if (auxDM) {
                    DMPlexPointLocalRef(auxDM, stencilInfo.cellId, locAuxArray, &auxPt) >> checkError;
                }

                // Get each of the stencil pts
                const PetscScalar *solStencilPt, *auxStencilPt = nullptr;
                DMPlexPointLocalRead(dm, stencilInfo.stencil.front(), localXArray, &solStencilPt) >> checkError;
                if (auxDM) {
                    DMPlexPointLocalRead(auxDM, stencilInfo.stencil.front(), locAuxArray, &auxStencilPt) >> checkError;
                }

                // update
                if (solPt) {
                    function.function(dim, &stencilInfo.geometry, cg, inputOffsetsPointer, solPt, solStencilPt, auxOffsetsPointer, auxPt, auxStencilPt, function.context) >> checkError;
                }
            }
        }

        // clean up access
        VecRestoreArray(subDomain->GetSolutionVector(), &globXArray) >> checkError;
        if (auto locAuxVec = subDomain->GetAuxVector()) {
            VecRestoreArray(locAuxVec, &locAuxArray) >> checkError;
        }

        // clean up the geom
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    }
    DMRestoreLocalVector(subDomain->GetDM(), &locXVec) >> checkError;
}

std::istream& ablate::boundarySolver::operator>>(std::istream& is, ablate::boundarySolver::BoundarySolver::BoundarySourceType& value) {
    std::string typeString;
    is >> typeString;
    ablate::utilities::StringUtilities::ToLower(typeString);

    if (typeString == "point")
        value = BoundarySolver::BoundarySourceType::Point;
    else if (typeString == "distributed")
        value = BoundarySolver::BoundarySourceType::Distributed;
    else if (typeString == "flux")
        value = BoundarySolver::BoundarySourceType::Flux;
    else if (typeString == "face")
        value = BoundarySolver::BoundarySourceType::Face;

    return is;
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::boundarySolver::BoundarySolver, "A solver used to compute boundary values in boundary cells", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::domain::Region, "fieldBoundary", "the region describing the faces between the boundary and field"),
         ARG(std::vector<ablate::boundarySolver::BoundaryProcess>, "processes", "a list of boundary processes"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         OPT(bool, "mergeFaces", "determine if multiple faces should be merged for a single cell, default if false"));

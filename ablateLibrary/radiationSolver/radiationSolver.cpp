//
// Created by owen on 3/19/22.
//
#include "radiationSolver.h"
#include <set>
#include <utility>
#include "radiationProcess.h"
#include "utilities/mathUtilities.hpp"

ablate::radiationSolver::RadiationSolver::RadiationSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> fieldBoundary,
                                                       std::vector<std::shared_ptr<RadiationProcess>> radiationProcesses, std::shared_ptr<parameters::Parameters> options)
    : CellSolver(std::move(solverId), std::move(region), std::move(options)), fieldBoundary(std::move(fieldBoundary)), radiationProcesses(std::move(radiationProcesses)) {}

ablate::radiationSolver::RadiationSolver::~RadiationSolver() {
    if (gradientCalculator) {
        PetscFVDestroy(&gradientCalculator); //We probably don't need anything related to gradients here?
    }
}

static void AddNeighborsToStencil(std::set<PetscInt>& stencilSet, DMLabel boundaryLabel, PetscInt boundaryValue, PetscInt depth, DM dm, PetscInt cell) {
    const PetscInt maxDepth = 2;

    // Check to see if this cell is already in the list
    if (stencilSet.count(cell)) {
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
                AddNeighborsToStencil(stencilSet, boundaryLabel, boundaryValue, depth + 1, dm, neighborCells[n]);
            }
        }
    }
}

void ablate::radiationSolver::RadiationSolver::Setup() { //allows initialization after the subdomain and dm is established
    // march over process and link to the flow
    for (auto& process : radiationProcesses) {
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
        throw std::invalid_argument("The RadiationSolver requires a PetscFVM that supports gradients.");
    }

    // Get the geometry for the mesh
    Vec faceGeomVec, cellGeomVec;
    DMPlexGetGeometryFVM(subDomain->GetDM(), &faceGeomVec, &cellGeomVec, nullptr) >> checkError;
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

    // Keep track of the current maxFaces
    PetscInt maxFaces = 0;

    // March over each cell in this region to create the stencil
    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // make sure we are not working on a ghost cell
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, cell, &ghost);
        }
        if (ghost >= 0) {
            // put in nan, should not be used
            gradientStencils.emplace_back(
                GradientStencil{.geometry = {.normal = {NAN, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}}, .stencil = {}, .gradientWeights = {}, .stencilSize = 0});
            continue;
        }

        // keep a list of cells in the stencil
        std::set<PetscInt> stencilSet{cell};

        // March over each face
        PetscInt numberFaces;
        const PetscInt* cellFaces;
        DMPlexGetConeSize(subDomain->GetDM(), cell, &numberFaces) >> checkError;
        DMPlexGetCone(subDomain->GetDM(), cell, &cellFaces) >> checkError;

        // Create a new BoundaryFVFaceGeom
        BoundaryFVFaceGeom geom{.normal = {0.0, 0.0, 0.0}, .areas = {0.0, 0.0, 0.0}, .centroid = {0.0, 0.0, 0.0}};

        // Set the face centroid to be equal to the face for gradient calc
        PetscFVCellGeom* cellGeom;
        DMPlexPointLocalRead(cellDM, cell, cellGeomArray, &cellGeom);
        PetscArraycpy(geom.centroid, cellGeom->centroid, dim) >> checkError;

        // Perform some error checking
        if (numberFaces < 1) {
            throw std::runtime_error("Isolated cell " + std::to_string(cell) + " cannot be used in RadiationSolver.");
        }

        // For each connected face
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

            for (PetscInt n = 0; n < numberNeighborCells; n++) {
                AddNeighborsToStencil(stencilSet, boundaryLabel, boundaryValue, 1, subDomain->GetDM(), neighborCells[n]);
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
                }
            } else {
                for (PetscInt d = 0; d < dim; d++) {
                    geom.normal[d] += fg->normal[d];
                    geom.areas[d] += fg->normal[d];
                }
            }
        }

        // compute the normal
        utilities::MathUtilities::NormVector(dim, geom.normal);

        // remove the boundary cell from the stencil
        stencilSet.erase(cell);

        // Compute the weights for the stencil
        std::vector<PetscInt> stencil(stencilSet.begin(), stencilSet.end());
        std::vector<PetscScalar> gradientWeights(stencil.size() * dim, 0.0);
        std::vector<PetscScalar> dx(stencil.size() * dim);
        std::vector<PetscScalar> volume(stencil.size());

        // Use a Reciprocal distance interpolate for the distribution weights.  This can be abstracted away in the future.
        std::vector<PetscScalar> distributionWeights(stencil.size(), 0.0);
        PetscScalar distributionWeightSum = 0.0;
        for (std::size_t n = 0; n < stencil.size(); n++) {
            PetscFVCellGeom* cg;
            DMPlexPointLocalRead(cellDM, stencil[n], cellGeomArray, &cg);
            for (PetscInt d = 0; d < dim; ++d) {
                dx[n * dim + d] = cg->centroid[d] - geom.centroid[d];
                distributionWeights[n] += PetscSqr(cg->centroid[d] - geom.centroid[d]);
            }
            distributionWeights[n] = 1.0 / PetscSqrtScalar(distributionWeights[n]);
            distributionWeightSum += distributionWeights[n];

            // store the volume
            volume[n] = cg->volume;
        }

        // normalize the distributionWeights
        utilities::MathUtilities::ScaleVector(distributionWeights.size(), distributionWeights.data(), 1.0 / distributionWeightSum);

        // Compute gradients
        if ((PetscInt)stencil.size() > maxFaces) {
            maxFaces = (PetscInt)stencil.size();
            PetscFVLeastSquaresSetMaxFaces(gradientCalculator, maxFaces) >> checkError;
        }
        PetscFVComputeGradient(gradientCalculator, (PetscInt)stencil.size(), &dx[0], &gradientWeights[0]) >> checkError;

        // Store the stencil
        gradientStencils.emplace_back(GradientStencil{
            .geometry = geom, .stencil = stencil, .gradientWeights = gradientWeights, .stencilSize = (PetscInt)stencil.size(), .distributionWeights = distributionWeights, .volumes = volume});

        maximumStencilSize = PetscMax(maximumStencilSize, (PetscInt)stencil.size());
    }
    RestoreRange(cellIS, cStart, cEnd, cells);

    // clean up the geom
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
}
void ablate::radiationSolver::RadiationSolver::Initialize() { //TODO: Initialization of the radiation domain needs to get the set of cells associated with each ray vector
    RegisterPreStage([](auto ts, auto& solver, auto stageTime) { //Adds function to be called before each flow stage
        Vec locXVec; //TODO: Need to create set of vectors associated with each ray in the initialization
        DMGetLocalVector(solver.GetSubDomain().GetDM(), &locXVec) >> checkError; //TODO: Need to find out what this function does, allows grabbing points at each location?
        DMGlobalToLocal(solver.GetSubDomain().GetDM(), solver.GetSubDomain().GetSolutionVector(), INSERT_VALUES, locXVec) >> checkError;

        // Get the time from the ts
        PetscReal time; //Radiation solver does not directly integrate in time. However, this may still be useful.
        TSGetTime(ts, &time) >> checkError;

        auto& cellSolver = dynamic_cast<CellSolver&>(solver);
        cellSolver.UpdateAuxFields(time, locXVec, solver.GetSubDomain().GetAuxVector()); ///This is the function that updates aux fields!

        DMRestoreLocalVector(solver.GetSubDomain().GetDM(), &locXVec) >> checkError;
    });
}
void ablate::radiationSolver::RadiationSolver::RegisterFunction(ablate::radiationSolver::RadiationSolver::BoundarySourceFunction function, void* context, const std::vector<std::string>& sourceFields,
                                                              const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields, BoundarySourceType type) {
    // Create the FVMRHS Function
    BoundaryFunctionDescription functionDescription{.function = function, .context = context, .type = type};

    for (auto& sourceField : sourceFields) {
        auto& fieldId = subDomain->GetField(sourceField); //Source field may not be useful or necessary for ray tracing
        functionDescription.sourceFields.push_back(fieldId.subId);
    }

    for (auto& inputField : inputFields) {
        auto& inputFieldId = subDomain->GetField(inputField);
        functionDescription.inputFields.push_back(inputFieldId.subId);
    }

    for (const auto& auxField : auxFields) { //Aux fields are an input to the RegisterFunction method //TODO: Need aux filed for temperature and absorptivity
        auto& auxFieldId = subDomain->GetField(auxField);
        functionDescription.auxFields.push_back(auxFieldId.subId);
    }

    boundaryFunctions.push_back(functionDescription);
}
PetscErrorCode ablate::radiationSolver::RadiationSolver::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) { //main interface for integrating in time Inputs: local vector, x vector (current solution), local f vector
    PetscFunctionBeginUser;                                                                                             //gets fields out of the main vector
    //TODO: This is likely where the radiative gain will be calculated and updated. The aux fields need to be updated and rays cast. Tracing vectors are precalculated in init.
    PetscErrorCode ierr;

    // Extract the cell geometry, and the dm that holds the information
    auto dm = subDomain->GetDM();
    auto auxDM = subDomain->GetAuxDM();
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;
    ierr = DMPlexGetGeometryFVM(dm, nullptr, &cellGeomVec, nullptr);
    CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCell);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);
    CHKERRQ(ierr);

    // prepare to compute the source, u, and a offsets
    PetscInt nf;
    ierr = PetscDSGetNumFields(subDomain->GetDiscreteSystem(), &nf);
    CHKERRQ(ierr);

    // Create the required offset arrays. These are sized for the max possible value
    PetscInt* offsetsTotal;
    ierr = PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &offsetsTotal);
    CHKERRQ(ierr);
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

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;
    // Get the region to march over
    IS cellIS; //(includes what cells are needed in the solver, iterate through, get needed values)
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);
    if (cEnd > cStart) {
        PetscInt dim = subDomain->GetDimensions();

        // Get pointers to sol, aux, and f vectors
        const PetscScalar *locXArray, *locAuxArray = nullptr;
        PetscScalar* locFArray;
        ierr = VecGetArrayRead(locXVec, &locXArray);
        CHKERRQ(ierr);
        if (auto locAuxVec = subDomain->GetAuxVector()) {
            ierr = VecGetArrayRead(locAuxVec, &locAuxArray);
            CHKERRQ(ierr);
        }
        VecGetArray(locFVec, &locFArray) >> checkError;
        CHKERRQ(ierr);

        // Store pointers to the stencil variables
        std::vector<const PetscScalar*> inputStencilValues(maximumStencilSize);
        std::vector<const PetscScalar*> auxStencilValues(maximumStencilSize);

        // March over each boundary function
        for (const auto& function : boundaryFunctions) {
            for (std::size_t i = 0; i < function.sourceFields.size(); i++) {
                sourceOffsets[i] = offsetsTotal[function.sourceFields[i]];
            }
            for (std::size_t i = 0; i < function.inputFields.size(); i++) {
                inputOffsets[i] = offsetsTotal[function.inputFields[i]];
            }
            for (std::size_t i = 0; i < function.auxFields.size(); i++) {
                auxOffsets[i] = auxOffTotal[function.inputFields[i]];
            }

            auto sourceOffsetsPointer = sourceOffsets.data();
            auto inputOffsetsPointer = inputOffsets.data();
            auto auxOffsetsPointer = auxOffsets.data();

            // March over each cell in this region
            PetscInt cOffset = 0;  // Keep track of the cell offset
            for (PetscInt c = cStart; c < cEnd; ++c, cOffset++) {
                // if there is a cell array, use it, otherwise it is just c
                const PetscInt cell = cells ? cells[c] : c;

                // make sure we are not working on a ghost cell
                PetscInt ghost = -1;
                if (ghostLabel) {
                    DMLabelGetValue(ghostLabel, cell, &ghost);
                }
                if (ghost >= 0) {
                    continue;
                }

                // Get the cell geom
                const PetscFVCellGeom* cg;
                DMPlexPointLocalRead(dmCell, cell, cellGeomArray, &cg) >> checkError;

                // Get pointers to the area of interest
                const PetscScalar *solPt, *auxPt = nullptr;
                DMPlexPointLocalRead(dm, cell, locXArray, &solPt) >> checkError;
                if (auxDM) {
                    DMPlexPointLocalRead(auxDM, cell, locAuxArray, &auxPt) >> checkError;
                }

                // Get each of the stencil pts
                const auto& stencilInfo = gradientStencils[cOffset];
                for (PetscInt p = 0; p < stencilInfo.stencilSize; p++) {
                    DMPlexPointLocalRead(dm, stencilInfo.stencil[p], locXArray, &inputStencilValues[p]) >> checkError;
                    if (auxDM) {
                        DMPlexPointLocalRead(auxDM, stencilInfo.stencil[p], locAuxArray, &auxStencilValues[p]) >> checkError;
                    }
                }

                // Get the pointer to the rhs
                switch (function.type) {
                    case BoundarySourceType::Point:
                        PetscScalar* rhs;
                        DMPlexPointLocalRef(dm, cell, locFArray, &rhs) >> checkError;

                        /*PetscErrorCode (*)(PetscInt dim, const BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell,
                                           const PetscInt uOff[], const PetscScalar* boundaryValues, const PetscScalar* stencilValues[],
                                           const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                           PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[], void* ctx)*/
                        ierr = function.function(dim,
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
                                                 function.context);
                        break;
                    case BoundarySourceType::Distributed:
                        /*PetscErrorCode (*)(PetscInt dim, const BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell,
                                                                   const PetscInt uOff[], const PetscScalar* boundaryValues, const PetscScalar* stencilValues[],
                                                                   const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                                                   PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[],
                           void* ctx)*/

                        // zero out the distributedSourceScratch
                        PetscArrayzero(distributedSourceScratch.data(), (PetscInt)distributedSourceScratch.size()) >> checkError;

                        ierr = function.function(dim,
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
                                                 function.context);

                        // Now distribute to each stencil point
                        for (PetscInt s = 0; s < stencilInfo.stencilSize; ++s) {
                            // Get the point in the rhs for this point.  It might be ghost but that is ok, the values are added together later
                            DMPlexPointLocalRef(dm, stencilInfo.stencil[s], locFArray, &rhs) >> checkError;

                            // Now over the entire rhs, the function should have added the values correctly using the sourceOffsetsPointer
                            for (PetscInt sc = 0; sc < scratchSize; sc++) {
                                rhs[sc] += (distributedSourceScratch[sc] * stencilInfo.distributionWeights[s]) / stencilInfo.volumes[s];
                            }
                        }

                        break;
                }

                CHKERRQ(ierr);
            }
        }

        // clean up access
        ierr = VecRestoreArrayRead(locXVec, &locXArray);
        CHKERRQ(ierr);
        if (auto locAuxVec = subDomain->GetAuxVector()) {
            ierr = VecRestoreArrayRead(locAuxVec, &locXArray);
            CHKERRQ(ierr);
        }
        VecRestoreArray(locFVec, &locFArray) >> checkError;
        CHKERRQ(ierr);

        // clean up the geom
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    }

    RestoreRange(cellIS, cStart, cEnd, cells);

    PetscFunctionReturn(0);
}
void ablate::radiationSolver::RadiationSolver::InsertFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, PetscReal time) {
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
        IS cellIS;
        PetscInt cStart, cEnd;
        const PetscInt* cells;
        GetCellRange(cellIS, cStart, cEnd, cells);
        PetscInt dim = subDomain->GetDimensions();

        // Get the petscFunction/context
        auto petscFunction = fieldFunction->GetFieldFunction()->GetPetscFunction();
        auto context = fieldFunction->GetFieldFunction()->GetContext();

        // March over each cell in this region
        PetscInt cOffset = 0;  // Keep track of the cell offset
        for (PetscInt c = cStart; c < cEnd; ++c, cOffset++) {
            // if there is a cell array, use it, otherwise it is just c
            const PetscInt cell = cells ? cells[c] : c;

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

        RestoreRange(cellIS, cStart, cEnd, cells);
        VecRestoreArray(vec, &array);
    }
}

void ablate::radiationSolver::RadiationSolver::ComputeGradient(PetscInt dim, PetscScalar boundaryValue, PetscInt stencilSize, const PetscScalar* stencilValues, const PetscScalar* stencilWeights,
                                                             PetscScalar* grad) {
    PetscArrayzero(grad, dim);

    for (PetscInt c = 0; c < stencilSize; ++c) {
        PetscScalar delta = stencilValues[c] - boundaryValue;

        for (PetscInt d = 0; d < dim; ++d) {
            grad[d] += stencilWeights[c * dim + d] * delta;
        }
    }
}

void ablate::radiationSolver::RadiationSolver::ComputeGradientAlongNormal(PetscInt dim, const ablate::radiationSolver::RadiationSolver::BoundaryFVFaceGeom* fg, PetscScalar boundaryValue,
                                                                        PetscInt stencilSize, const PetscScalar* stencilValues, const PetscScalar* stencilWeights, PetscScalar& dPhiDNorm) {
    dPhiDNorm = 0.0;
    for (PetscInt c = 0; c < stencilSize; ++c) {
        PetscScalar delta = stencilValues[c] - boundaryValue;

        for (PetscInt d = 0; d < dim; ++d) {
            dPhiDNorm += stencilWeights[c * dim + d] * delta * fg->normal[d];
        }
    }
}

const ablate::radiationSolver::RadiationSolver::BoundaryFVFaceGeom& ablate::radiationSolver::RadiationSolver::GetBoundaryGeometry(PetscInt cell) const {
    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);

    // Locate the index
    PetscInt location;
    ISLocate(cellIS, cell, &location) >> checkError;
    RestoreRange(cellIS, cStart, cEnd, cells);

    return gradientStencils[location].geometry;
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::radiationSolver::RadiationSolver, "A solver used to compute boundary values in boundary cells", ARG(std::string, "id", "the name of the flow field"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(ablate::domain::Region, "fieldBoundary", "the region describing the faces between the boundary and field"),
         ARG(std::vector<ablate::radiationSolver::RadiationProcess>, "processes", "a list of boundary processes"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"));

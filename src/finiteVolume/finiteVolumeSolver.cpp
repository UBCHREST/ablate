#include "finiteVolumeSolver.hpp"
#include <petsc/private/dmpleximpl.h>

#include <utility>
#include "faceInterpolant.hpp"
#include "processes/process.hpp"
#include "utilities/mpiError.hpp"
#include "utilities/petscError.hpp"

ablate::finiteVolume::FiniteVolumeSolver::FiniteVolumeSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                             std::vector<std::shared_ptr<processes::Process>> processes,
                                                             std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions, bool computePhysicsTimeStep)
    : CellSolver(std::move(solverId), std::move(region), std::move(options)),
      computePhysicsTimeStep(computePhysicsTimeStep),
      processes(std::move(processes)),
      boundaryConditions(std::move(boundaryConditions)) {}

ablate::finiteVolume::FiniteVolumeSolver::~FiniteVolumeSolver() {
    for (auto& dm : gradientCellDms) {
        if (dm) {
            DMDestroy(&dm) >> checkError;
        }
    }
    for (auto& dm : auxGradientCellDms) {
        if (dm) {
            DMDestroy(&dm) >> checkError;
        }
    }
}

void ablate::finiteVolume::FiniteVolumeSolver::Setup() {
    ablate::solver::CellSolver::Setup();

    // march over process and link to the flow
    for (const auto& process : processes) {
        process->Initialize(*this);
    }

    // Set the flux calculator solver for each component
    PetscDSSetFromOptions(subDomain->GetDiscreteSystem()) >> checkError;

    // Some petsc code assumes that a ghostLabel has created, so create one
    PetscBool ghostLabel;
    DMHasLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;
    if (!ghostLabel) {
        throw std::runtime_error("The FiniteVolumeSolver expects ghost cells around the boundary even if the FiniteVolumeSolver region does not include the boundary.");
    }
}

void ablate::finiteVolume::FiniteVolumeSolver::Initialize() {
    // add each boundary condition
    for (const auto& boundary : boundaryConditions) {
        const auto& fieldId = subDomain->GetField(boundary->GetFieldName());

        // Setup the boundary condition
        boundary->SetupBoundary(subDomain->GetDM(), subDomain->GetDiscreteSystem(), fieldId.id);
    }

    // copy over any boundary information from the dm, to the aux dm and set the sideset
    if (subDomain->GetAuxDM()) {
        PetscDS flowProblem = subDomain->GetDiscreteSystem();
        PetscDS auxProblem = subDomain->GetAuxDiscreteSystem();

        // Get the number of boundary conditions and other info
        PetscInt numberBC;
        PetscDSGetNumBoundary(flowProblem, &numberBC) >> checkError;
        PetscInt numberAuxFields;
        PetscDSGetNumFields(auxProblem, &numberAuxFields) >> checkError;

        for (PetscInt bc = 0; bc < numberBC; bc++) {
            DMBoundaryConditionType type;
            const char* name;
            DMLabel label;
            PetscInt field;
            PetscInt numberIds;
            const PetscInt* ids;

            // Get the boundary
            PetscDSGetBoundary(flowProblem, bc, nullptr, &type, &name, &label, &numberIds, &ids, &field, nullptr, nullptr, nullptr, nullptr, nullptr) >> checkError;

            // If this is for euler and DM_BC_NATURAL_RIEMANN add it to the aux
            if (type == DM_BC_NATURAL_RIEMANN && field == 0) {
                for (PetscInt af = 0; af < numberAuxFields; af++) {
                    PetscDSAddBoundary(auxProblem, type, name, label, numberIds, ids, af, 0, nullptr, nullptr, nullptr, nullptr, nullptr) >> checkError;
                }
            }
        }
    }
    if (!timeStepFunctions.empty() && computePhysicsTimeStep) {
        RegisterPreStep(EnforceTimeStep);
    }

    auto getGradientDm = [this](const domain::Field& fieldInfo, std::vector<DM>& gradDMs) {
        auto petscField = subDomain->GetPetscFieldObject(fieldInfo);
        auto petscFieldFV = (PetscFV)petscField;

        PetscBool computeGradients;
        PetscFVGetComputeGradients(petscFieldFV, &computeGradients) >> checkError;

        if (computeGradients) {
            DM dmGradInt;

            DMLabel regionLabel = nullptr;
            PetscInt regionValue = PETSC_DECIDE;
            if (const auto& region = GetRegion()) {
                DMGetLabel(subDomain->GetDM(), region->GetName().c_str(), &regionLabel);
                regionValue = region->GetValue();
            }

            ComputeGradientFVM(subDomain->GetFieldDM(fieldInfo), regionLabel, regionValue, petscFieldFV, faceGeomVec, cellGeomVec, &dmGradInt) >> checkError;
            gradDMs.push_back(dmGradInt);
        } else {
            gradDMs.push_back(nullptr);
        }
    };

    // Compute the gradient dm for each field that supports it
    for (const auto& fieldInfo : subDomain->GetFields()) {
        getGradientDm(fieldInfo, gradientCellDms);
    }
    for (const auto& fieldInfo : subDomain->GetFields(domain::FieldLocation::AUX)) {
        getGradientDm(fieldInfo, auxGradientCellDms);
    }
}

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    auto dm = subDomain->GetDM();
    auto ds = subDomain->GetDiscreteSystem();
    /* Handle non-essential (e.g. outflow) boundary values.  This should be done before the auxFields are updated so that boundary values can be updated */
    ierr = ablate::solver::Solver::DMPlexInsertBoundaryValues_Plex(dm, ds, PETSC_FALSE, locXVec, time, faceGeomVec, cellGeomVec, nullptr);
    CHKERRQ(ierr);

    try {
        // update any aux fields, including ghost cells
        UpdateAuxFields(time, locXVec, subDomain->GetAuxVector());

        // Compute the RHS function
        ComputeCellSourceTerms(time, locXVec, subDomain->GetAuxVector(), locFVec);

        // create the faceInterpolant if needed
        if (!continuousFluxFunctionDescriptions.empty()) {
            if (faceInterpolant == nullptr) {
                faceInterpolant = std::make_unique<FaceInterpolant>(subDomain, faceGeomVec, cellGeomVec);
            }

            IS faceIS;
            PetscInt fStart, fEnd;
            const PetscInt* faces;
            GetFaceRange(faceIS, fStart, fEnd, faces);

            faceInterpolant->ComputeRHS(time, locXVec, subDomain->GetAuxVector(), locFVec, GetRegion(), continuousFluxFunctionDescriptions, fStart, fEnd, faces, cellGeomVec);
            RestoreRange(faceIS, fStart, fEnd, faces);
        }
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exception.what());
    }

    // iterate over any arbitrary RHS functions
    for (const auto& rhsFunction : rhsArbitraryFunctions) {
        ierr = rhsFunction.first(*this, dm, time, locXVec, locFVec, rhsFunction.second);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(DiscontinuousFluxFunction function, void* context, const std::string& field, const std::vector<std::string>& inputFields,
                                                                   const std::vector<std::string>& auxFields) {
    // map the field, inputFields, and auxFields to locations
    auto& fieldId = subDomain->GetField(field);

    // Create the FVMRHS Function
    DiscontinuousFluxFunctionDescription functionDescription{.function = function, .context = context, .field = fieldId.id};

    for (auto& inputField : inputFields) {
        auto& inputFieldId = subDomain->GetField(inputField);
        functionDescription.inputFields.push_back(inputFieldId.id);
    }

    for (const auto& auxField : auxFields) {
        auto& auxFieldId = subDomain->GetField(auxField);
        functionDescription.auxFields.push_back(auxFieldId.id);
    }

    discontinuousFluxFunctionDescriptions.push_back(functionDescription);
}

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(ablate::finiteVolume::FaceInterpolant::ContinuousFluxFunction function, void* context, const std::string& field,
                                                                   const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields) {
    // map the field, inputFields, and auxFields to locations
    auto& fieldId = subDomain->GetField(field);

    // Create the FVMRHS Function
    FaceInterpolant::ContinuousFluxFunctionDescription functionDescription{.function = function, .context = context, .field = fieldId.id};

    for (auto& inputField : inputFields) {
        auto& inputFieldId = subDomain->GetField(inputField);
        functionDescription.inputFields.push_back(inputFieldId.id);
    }

    for (const auto& auxField : auxFields) {
        auto& auxFieldId = subDomain->GetField(auxField);
        functionDescription.auxFields.push_back(auxFieldId.id);
    }

    continuousFluxFunctionDescriptions.push_back(functionDescription);
}

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(PointFunction function, void* context, const std::vector<std::string>& fields, const std::vector<std::string>& inputFields,
                                                                   const std::vector<std::string>& auxFields) {
    // Create the FVMRHS Function
    PointFunctionDescription functionDescription{.function = function, .context = context};

    for (const auto& field : fields) {
        auto& fieldId = subDomain->GetField(field);
        functionDescription.fields.push_back(fieldId.id);
    }

    for (const auto& inputField : inputFields) {
        auto& fieldId = subDomain->GetField(inputField);
        functionDescription.inputFields.push_back(fieldId.id);
    }

    for (const auto& auxField : auxFields) {
        auto& fieldId = subDomain->GetField(auxField);
        functionDescription.auxFields.push_back(fieldId.id);
    }

    pointFunctionDescriptions.push_back(functionDescription);
}

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(RHSArbitraryFunction function, void* context) { rhsArbitraryFunctions.emplace_back(function, context); }

void ablate::finiteVolume::FiniteVolumeSolver::EnforceTimeStep(TS ts, ablate::solver::Solver& solver) {
    auto& flowFV = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver&>(solver);
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> checkError;
    PetscInt timeStep;
    TSGetStepNumber(ts, &timeStep) >> checkError;
    PetscReal currentDt;
    TSGetTimeStep(ts, &currentDt) >> checkError;

    // march over each calculator
    PetscReal dtMin = 1000.0;
    for (const auto& dtFunction : flowFV.timeStepFunctions) {
        dtMin = PetscMin(dtMin, dtFunction.function(ts, flowFV, dtFunction.context));
    }

    // take the min across all ranks
    PetscMPIInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);

    PetscReal dtMinGlobal;
    MPI_Allreduce(&dtMin, &dtMinGlobal, 1, MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)ts)) >> checkMpiError;

    // don't override the first time step if bigger
    if (timeStep > 0 || dtMinGlobal < currentDt) {
        TSSetTimeStep(ts, dtMinGlobal) >> checkError;
        if (PetscIsNanReal(dtMinGlobal)) {
            throw std::runtime_error("Invalid timestep selected for flow");
        }
    }
}

void ablate::finiteVolume::FiniteVolumeSolver::RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx, std::string name) {
    timeStepFunctions.emplace_back(ComputeTimeStepDescription{.function = function, .context = ctx, .name = std::move(name)});
}

void ablate::finiteVolume::FiniteVolumeSolver::ComputeCellSourceTerms(PetscReal time, Vec locXVec, Vec locAuxField, Vec locF) {
    auto dm = subDomain->GetDM();
    auto dmAux = subDomain->GetAuxDM();

    /* 1: Get sizes from dm and dmAux */
    PetscSection section = nullptr;
    DMGetLocalSection(dm, &section) >> checkError;

    // Get the ds from he subDomain and required info
    auto ds = subDomain->GetDiscreteSystem();
    PetscInt nf, totDim;
    PetscDSGetNumFields(ds, &nf) >> checkError;
    PetscDSGetTotalDimension(ds, &totDim) >> checkError;

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    PetscDS dsAux = subDomain->GetAuxDiscreteSystem();
    PetscInt naf = 0, totDimAux = 0;
    if (locAuxField) {
        PetscDSGetTotalDimension(dsAux, &totDimAux) >> checkError;
        PetscDSGetNumFields(dsAux, &naf) >> checkError;
    }

    /* 2: Get geometric data */
    // We can use a single call for the geometry data because it does not depend on the fv object
    const PetscScalar* cellGeomArray = nullptr;
    const PetscScalar* faceGeomArray = nullptr;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
    DM faceDM, cellDM;
    VecGetDM(faceGeomVec, &faceDM) >> checkError;
    VecGetDM(cellGeomVec, &cellDM) >> checkError;

    // Get raw access to the computed values
    const PetscScalar *xArray, *auxArray = nullptr;
    VecGetArrayRead(locXVec, &xArray) >> checkError;
    if (locAuxField) {
        VecGetArrayRead(locAuxField, &auxArray) >> checkError;
    }

    // get raw access to the locF
    PetscScalar* locFArray;
    VecGetArray(locF, &locFArray) >> checkError;

    // Compute the source terms from flux across the interface for cell based gradient functions
    if (!this->discontinuousFluxFunctionDescriptions.empty() || !this->pointFunctionDescriptions.empty()) {
        // there must be a separate gradient vector/dm for field because they can be different sizes
        std::vector<Vec> locGradVecs(nf, nullptr);
        std::vector<Vec> locAuxGradVecs(naf, nullptr);

        /* Reconstruct and limit cell gradients */
        // for each field compute the gradient in the localGrads vector
        for (const auto& field : subDomain->GetFields()) {
            ComputeFieldGradients(field, locXVec, locGradVecs[field.subId], gradientCellDms[field.subId]);
        }

        // do the same for the aux fields
        for (const auto& field : subDomain->GetFields(domain::FieldLocation::AUX)) {
            ComputeFieldGradients(field, locAuxField, locAuxGradVecs[field.subId], auxGradientCellDms[field.subId]);

            if (auxGradientCellDms[field.subId]) {
                auto fvm = (PetscFV)subDomain->GetPetscFieldObject(field);
                FillGradientBoundary(dmAux, fvm, locAuxField, locAuxGradVecs[field.subId]) >> checkError;
            }
        }

        std::vector<const PetscScalar*> locGradArrays(nf, nullptr);
        std::vector<const PetscScalar*> locAuxGradArrays(naf, nullptr);
        for (const auto& field : subDomain->GetFields()) {
            if (locGradVecs[field.subId]) {
                VecGetArrayRead(locGradVecs[field.subId], &locGradArrays[field.subId]) >> checkError;
            }
        }
        for (const auto& field : subDomain->GetFields(domain::FieldLocation::AUX)) {
            if (locAuxGradVecs[field.subId]) {
                VecGetArrayRead(locAuxGradVecs[field.subId], &locAuxGradArrays[field.subId]) >> checkError;
            }
        }

        if (!this->discontinuousFluxFunctionDescriptions.empty()) {
            ComputeFluxSourceTerms(dm,
                                   ds,
                                   totDim,
                                   xArray,
                                   dmAux,
                                   dsAux,
                                   totDimAux,
                                   auxArray,
                                   faceDM,
                                   faceGeomArray,
                                   cellDM,
                                   cellGeomArray,
                                   gradientCellDms,
                                   locGradArrays,
                                   auxGradientCellDms,
                                   locAuxGradArrays,
                                   locFArray);
        }
        if (!this->pointFunctionDescriptions.empty()) {
            ComputePointSourceTerms(dm,
                                    ds,
                                    totDim,
                                    time,
                                    xArray,
                                    dmAux,
                                    dsAux,
                                    totDimAux,
                                    auxArray,
                                    faceDM,
                                    faceGeomArray,
                                    cellDM,
                                    cellGeomArray,
                                    gradientCellDms,
                                    locGradArrays,
                                    auxGradientCellDms,
                                    locAuxGradArrays,
                                    locFArray);
        }

        // clean up cell grads
        for (const auto& field : subDomain->GetFields()) {
            if (locGradVecs[field.subId]) {
                VecRestoreArrayRead(locGradVecs[field.subId], &locGradArrays[field.subId]) >> checkError;
                DMRestoreLocalVector(gradientCellDms[field.subId], &locGradVecs[field.subId]) >> checkError;
            }
        }
        for (const auto& field : subDomain->GetFields(domain::FieldLocation::AUX)) {
            if (locAuxGradVecs[field.subId]) {
                VecRestoreArrayRead(locAuxGradVecs[field.subId], &locAuxGradArrays[field.subId]) >> checkError;
                DMRestoreLocalVector(auxGradientCellDms[field.subId], &locAuxGradVecs[field.subId]) >> checkError;
            }
        }
    }

    // cleanup (restore access to locGradVecs, locAuxGradVecs with DMRestoreLocalVector)
    VecRestoreArrayRead(locXVec, &xArray) >> checkError;
    if (locAuxField) {
        VecRestoreArrayRead(locAuxField, &auxArray) >> checkError;
    }

    VecRestoreArray(locF, &locFArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, (const PetscScalar**)&cellGeomArray) >> checkError;
    VecRestoreArrayRead(cellGeomVec, (const PetscScalar**)&faceGeomArray) >> checkError;
}

/**
 * This is a duplication of PETSC that we don't have access to
 */
static PetscErrorCode DMPlexApplyLimiter_Internal(DM dm, DM dmCell, PetscLimiter lim, PetscInt dim, PetscInt dof, PetscInt cell, PetscInt field, PetscInt face, PetscInt fStart, PetscInt fEnd,
                                                  PetscReal* cellPhi, const PetscScalar* x, const PetscScalar* cellgeom, const PetscFVCellGeom* cg, const PetscScalar* cx, const PetscScalar* cgrad) {
    const PetscInt* children;
    PetscInt numChildren;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = DMPlexGetTreeChildren(dm, face, &numChildren, &children);
    CHKERRQ(ierr);
    if (numChildren) {
        PetscInt c;

        for (c = 0; c < numChildren; c++) {
            PetscInt childFace = children[c];

            if (childFace >= fStart && childFace < fEnd) {
                ierr = DMPlexApplyLimiter_Internal(dm, dmCell, lim, dim, dof, cell, field, childFace, fStart, fEnd, cellPhi, x, cellgeom, cg, cx, cgrad);
                CHKERRQ(ierr);
            }
        }
    } else {
        PetscScalar* ncx;
        PetscFVCellGeom* ncg;
        const PetscInt* fcells;
        PetscInt ncell, d;
        PetscReal v[3];

        ierr = DMPlexGetSupport(dm, face, &fcells);
        CHKERRQ(ierr);
        ncell = cell == fcells[0] ? fcells[1] : fcells[0];
        if (field >= 0) {
            ierr = DMPlexPointLocalFieldRead(dm, ncell, field, x, &ncx);
            CHKERRQ(ierr);
        } else {
            ierr = DMPlexPointLocalRead(dm, ncell, x, &ncx);
            CHKERRQ(ierr);
        }
        ierr = DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg);
        CHKERRQ(ierr);
        DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, ncg->centroid, v);
        for (d = 0; d < dof; ++d) {
            /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
            PetscReal denom = DMPlex_DotD_Internal(dim, &cgrad[d * dim], v);
            PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / denom;

            ierr = PetscLimiterLimit(lim, flim, &phi);
            CHKERRQ(ierr);
            cellPhi[d] = PetscMin(cellPhi[d], phi);
        }
    }
    PetscFunctionReturn(0);
}

void ablate::finiteVolume::FiniteVolumeSolver::ComputeFieldGradients(const ablate::domain::Field& field, Vec xLocalVec, Vec& gradLocVec, DM& dmGrad) {
    // get the FVM petsc field associated with this field
    auto fvm = (PetscFV)subDomain->GetPetscFieldObject(field);
    auto dm = subDomain->GetFieldDM(field);

    // Get the dm for this grad field
    // If there is no grad, return
    if (!dmGrad) {
        return;
    }

    // Create a gradLocVec
    DMGetLocalVector(dmGrad, &gradLocVec) >> checkError;

    // Get the correct sized vec (gradient for this field)
    Vec gradGlobVec;
    DMGetGlobalVector(dmGrad, &gradGlobVec) >> checkError;
    VecZeroEntries(gradGlobVec) >> checkError;

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

    // Get the face geometry
    DM dmFace;
    const PetscScalar* faceGeometryArray;
    VecGetDM(faceGeomVec, &dmFace) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeometryArray);

    // extract the local x array
    const PetscScalar* xLocalArray;
    VecGetArrayRead(xLocalVec, &xLocalArray);

    // extract the global grad array
    PetscScalar* gradGlobArray;
    VecGetArray(gradGlobVec, &gradGlobArray);

    // March over only the faces in region
    IS faceIS;
    PetscInt fStart, fEnd;
    const PetscInt* faces;
    GetFaceRange(faceIS, fStart, fEnd, faces);

    // Get the dof and dim
    PetscInt dim = subDomain->GetDimensions();
    PetscInt dof = field.numberComponents;

    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscInt face = faces ? faces[f] : f;

        // make sure that this is a face we should use
        PetscBool boundary;
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, face, &ghost);
        }
        DMIsBoundaryPoint(dm, face, &boundary);
        PetscInt numChildren;
        DMPlexGetTreeChildren(dm, face, &numChildren, nullptr);
        if (ghost >= 0 || boundary || numChildren) continue;

        // Do a sanity check on the number of cells connected to this face
        PetscInt numCells;
        DMPlexGetSupportSize(dm, face, &numCells);
        if (numCells != 2) {
            throw std::runtime_error("face " + std::to_string(face) + " has " + std::to_string(numCells) + " support points (cells): expected 2");
        }

        // add in the contributions from this face
        const PetscInt* cells;
        PetscFVFaceGeom* fg;
        PetscScalar* cx[2];
        PetscScalar* cgrad[2];

        DMPlexGetSupport(dm, face, &cells);
        DMPlexPointLocalRead(dmFace, face, faceGeometryArray, &fg);
        for (PetscInt c = 0; c < 2; ++c) {
            DMPlexPointLocalFieldRead(dm, cells[c], field.id, xLocalArray, &cx[c]) >> checkError;
            DMPlexPointGlobalRef(dmGrad, cells[c], gradGlobArray, &cgrad[c]) >> checkError;
        }
        for (PetscInt pd = 0; pd < dof; ++pd) {
            PetscScalar delta = cx[1][pd] - cx[0][pd];

            for (PetscInt d = 0; d < dim; ++d) {
                if (cgrad[0]) cgrad[0][pd * dim + d] += fg->grad[0][d] * delta;
                if (cgrad[1]) cgrad[1][pd * dim + d] -= fg->grad[1][d] * delta;
            }
        }
    }

    // Check for a limiter the limiter
    PetscLimiter lim;
    PetscFVGetLimiter(fvm, &lim) >> checkError;
    if (lim) {
        /* Limit interior gradients (using cell-based loop because it generalizes better to vector limiters) */
        IS cellIS;
        PetscInt cStart, cEnd;
        const PetscInt* cells;
        GetCellRange(cellIS, cStart, cEnd, cells);

        // Get the cell geometry
        DM dmCell;
        const PetscScalar* cellGeometryArray;
        VecGetDM(cellGeomVec, &dmCell) >> checkError;
        VecGetArrayRead(cellGeomVec, &cellGeometryArray);

        // create a temp work array
        PetscReal* cellPhi;
        DMGetWorkArray(dm, dof, MPIU_REAL, &cellPhi) >> checkError;

        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscInt cell = cells ? cells[c] : c;

            const PetscInt* cellFaces;
            PetscScalar* cx;
            PetscFVCellGeom* cg;
            PetscScalar* cgrad;
            PetscInt coneSize;

            DMPlexGetConeSize(dm, cell, &coneSize) >> checkError;
            DMPlexGetCone(dm, cell, &cellFaces) >> checkError;
            DMPlexPointLocalFieldRead(dm, cell, field.id, xLocalArray, &cx) >> checkError;
            DMPlexPointLocalRead(dmCell, cell, cellGeometryArray, &cg) >> checkError;
            DMPlexPointGlobalRef(dmGrad, cell, gradGlobArray, &cgrad) >> checkError;

            if (!cgrad) {
                /* Unowned overlap cell, we do not compute */
                continue;
            }
            /* Limiter will be minimum value over all neighbors */
            for (PetscInt d = 0; d < dof; ++d) {
                cellPhi[d] = PETSC_MAX_REAL;
            }
            for (PetscInt f = 0; f < coneSize; ++f) {
                DMPlexApplyLimiter_Internal(dm, dmCell, lim, dim, dof, cell, field.id, cellFaces[f], fStart, fEnd, cellPhi, xLocalArray, cellGeometryArray, cg, cx, cgrad) >> checkError;
            }
            /* Apply limiter to gradient */
            for (PetscInt pd = 0; pd < dof; ++pd) {
                /* Scalar limiter applied to each component separately */
                for (PetscInt d = 0; d < dim; ++d) {
                    cgrad[pd * dim + d] *= cellPhi[pd];
                }
            }
        }

        // clean up the limiter work
        DMRestoreWorkArray(dm, dof, MPIU_REAL, &cellPhi) >> checkError;
        RestoreRange(cellIS, cStart, cEnd, cells);
        VecRestoreArrayRead(cellGeomVec, &cellGeometryArray);
    }
    // Communicate gradient values
    VecRestoreArray(gradGlobVec, &gradGlobArray) >> checkError;
    DMGlobalToLocalBegin(dmGrad, gradGlobVec, INSERT_VALUES, gradLocVec) >> checkError;
    DMGlobalToLocalEnd(dmGrad, gradGlobVec, INSERT_VALUES, gradLocVec) >> checkError;

    // cleanup
    VecRestoreArrayRead(xLocalVec, &xLocalArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeometryArray) >> checkError;
    RestoreRange(faceIS, fStart, fEnd, faces);
    DMRestoreGlobalVector(dmGrad, &gradGlobVec) >> checkError;
}

void ablate::finiteVolume::FiniteVolumeSolver::ProjectToFace(const std::vector<domain::Field>& fields, PetscDS ds, const PetscFVFaceGeom& faceGeom, PetscInt cellId, const PetscFVCellGeom& cellGeom,
                                                             DM dm, const PetscScalar* xArray, const std::vector<DM>& dmGrads, const std::vector<const PetscScalar*>& gradArrays, PetscScalar* u,
                                                             PetscScalar* grad, bool projectField) {
    const auto dim = subDomain->GetDimensions();

    // Keep track of derivative offset
    PetscInt* offsets;
    PetscInt* dirOffsets;
    PetscDSGetComponentOffsets(ds, &offsets) >> checkError;
    PetscDSGetComponentDerivativeOffsets(ds, &dirOffsets) >> checkError;

    // March over each field
    for (const auto& field : fields) {
        PetscReal dx[3];
        PetscScalar* xCell;
        PetscScalar* gradCell;

        // Get the field values at this cell
        DMPlexPointLocalFieldRead(dm, cellId, field.subId, xArray, &xCell) >> checkError;

        // If we need to project the field
        if (projectField && dmGrads[field.subId]) {
            DMPlexPointLocalRead(dmGrads[field.subId], cellId, gradArrays[field.subId], &gradCell) >> checkError;
            DMPlex_WaxpyD_Internal(dim, -1, cellGeom.centroid, faceGeom.centroid, dx);

            // Project the cell centered value onto the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c] + DMPlex_DotD_Internal(dim, &gradCell[c * dim], dx);

                // copy the gradient into the grad vector
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = gradCell[c * dim + d];
                }
            }

        } else if (dmGrads[field.subId]) {
            // Project the cell centered value onto the face
            DMPlexPointLocalRead(dmGrads[field.subId], cellId, gradArrays[field.subId], &gradCell) >> checkError;
            // Project the cell centered value onto the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c];

                // copy the gradient into the grad vector
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = gradCell[c * dim + d];
                }
            }

        } else {
            // Just copy the cell centered value on to the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c];

                // fill the grad with NAN to prevent use
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = NAN;
                }
            }
        }
    }
}

void ablate::finiteVolume::FiniteVolumeSolver::ComputeFluxSourceTerms(DM dm, PetscDS ds, PetscInt totDim, const PetscScalar* xArray, DM dmAux, PetscDS dsAux, PetscInt totDimAux,
                                                                      const PetscScalar* auxArray, DM faceDM, const PetscScalar* faceGeomArray, DM cellDM, const PetscScalar* cellGeomArray,
                                                                      std::vector<DM>& dmGrads, std::vector<const PetscScalar*>& locGradArrays, std::vector<DM>& dmAuxGrads,
                                                                      std::vector<const PetscScalar*>& locAuxGradArrays, PetscScalar* locFArray) {
    PetscInt dim = subDomain->GetDimensions();

    // Size up the work arrays (uL, uR, gradL, gradR, auxL, auxR, gradAuxL, gradAuxR), these are only sized for one face at a time
    PetscScalar* flux;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &flux) >> checkError;

    PetscScalar *uL, *uR;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &uL) >> checkError;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &uR) >> checkError;

    PetscScalar *gradL, *gradR;
    DMGetWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradL) >> checkError;
    DMGetWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradR) >> checkError;

    // size up the aux variables
    PetscScalar *auxL = nullptr, *auxR = nullptr;
    PetscScalar *gradAuxL = nullptr, *gradAuxR = nullptr;
    if (dmAux) {
        DMGetWorkArray(dmAux, totDimAux, MPIU_SCALAR, &auxL) >> checkError;
        DMGetWorkArray(dmAux, totDimAux, MPIU_SCALAR, &auxR) >> checkError;

        DMGetWorkArray(dmAux, dim * totDimAux, MPIU_SCALAR, &gradAuxR) >> checkError;
        DMGetWorkArray(dmAux, dim * totDimAux, MPIU_SCALAR, &gradAuxL) >> checkError;
    }

    // Precompute the offsets to pass into the rhsFluxFunctionDescriptions
    std::vector<PetscInt> fluxComponentSize(discontinuousFluxFunctionDescriptions.size());
    std::vector<PetscInt> fluxId(discontinuousFluxFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> uOff(discontinuousFluxFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> aOff(discontinuousFluxFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> uOff_x(discontinuousFluxFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> aOff_x(discontinuousFluxFunctionDescriptions.size());

    // Get the full set of offsets from the ds
    PetscInt* uOffTotal;
    PetscInt* uGradOffTotal;
    PetscDSGetComponentOffsets(ds, &uOffTotal) >> checkError;
    PetscDSGetComponentDerivativeOffsets(ds, &uGradOffTotal) >> checkError;

    for (std::size_t fun = 0; fun < discontinuousFluxFunctionDescriptions.size(); fun++) {
        const auto& field = subDomain->GetField(discontinuousFluxFunctionDescriptions[fun].field);
        fluxComponentSize[fun] = field.numberComponents;
        fluxId[fun] = field.id;
        for (std::size_t f = 0; f < discontinuousFluxFunctionDescriptions[fun].inputFields.size(); f++) {
            uOff[fun].push_back(uOffTotal[discontinuousFluxFunctionDescriptions[fun].inputFields[f]]);
            uOff_x[fun].push_back(uGradOffTotal[discontinuousFluxFunctionDescriptions[fun].inputFields[f]]);
        }
    }

    if (dsAux) {
        PetscInt* auxOffTotal;
        PetscInt* auxGradOffTotal;
        PetscDSGetComponentOffsets(dsAux, &auxOffTotal) >> checkError;
        PetscDSGetComponentDerivativeOffsets(dsAux, &auxGradOffTotal) >> checkError;
        for (std::size_t fun = 0; fun < discontinuousFluxFunctionDescriptions.size(); fun++) {
            for (std::size_t f = 0; f < discontinuousFluxFunctionDescriptions[fun].auxFields.size(); f++) {
                aOff[fun].push_back(auxOffTotal[discontinuousFluxFunctionDescriptions[fun].auxFields[f]]);
                aOff_x[fun].push_back(auxGradOffTotal[discontinuousFluxFunctionDescriptions[fun].auxFields[f]]);
            }
        }
    }
    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

    // get the label for this region
    DMLabel regionLabel = nullptr;
    PetscInt regionValue = 0;
    if (auto region = GetRegion()) {
        regionValue = region->GetValue();
        DMGetLabel(dm, region->GetName().c_str(), &regionLabel) >> checkError;
    }

    // March over each face in this region
    IS faceIS;
    PetscInt fStart, fEnd;
    const PetscInt* faces;
    GetFaceRange(faceIS, fStart, fEnd, faces);
    for (PetscInt f = fStart; f < fEnd; ++f) {
        const PetscInt face = faces ? faces[f] : f;

        // make sure that this is a valid face
        PetscInt ghost, nsupp, nchild;
        DMLabelGetValue(ghostLabel, face, &ghost) >> checkError;
        DMPlexGetSupportSize(dm, face, &nsupp) >> checkError;
        DMPlexGetTreeChildren(dm, face, &nchild, nullptr) >> checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

        // Get the face geometry
        const PetscInt* faceCells;
        PetscFVFaceGeom* fg;
        PetscFVCellGeom *cgL, *cgR;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;
        DMPlexGetSupport(dm, face, &faceCells) >> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[0], cellGeomArray, &cgL) >> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[1], cellGeomArray, &cgR) >> checkError;

        PetscInt leftFlowLabelValue = regionValue;
        PetscInt rightFlowLabelValue = regionValue;
        if (regionLabel) {
            DMLabelGetValue(regionLabel, faceCells[0], &leftFlowLabelValue);
            DMLabelGetValue(regionLabel, faceCells[1], &rightFlowLabelValue);
        }
        // compute the left/right face values
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[0], *cgL, dm, xArray, dmGrads, locGradArrays, uL, gradL, leftFlowLabelValue == regionValue);
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[1], *cgR, dm, xArray, dmGrads, locGradArrays, uR, gradR, rightFlowLabelValue == regionValue);

        // determine the left/right cells
        if (auxArray) {
            ProjectToFace(subDomain->GetFields(domain::FieldLocation::AUX), dsAux, *fg, faceCells[0], *cgL, dmAux, auxArray, dmAuxGrads, locAuxGradArrays, auxL, gradAuxL, false);
            ProjectToFace(subDomain->GetFields(domain::FieldLocation::AUX), dsAux, *fg, faceCells[1], *cgR, dmAux, auxArray, dmAuxGrads, locAuxGradArrays, auxR, gradAuxR, false);
        }

        // March over each source function
        for (std::size_t fun = 0; fun < discontinuousFluxFunctionDescriptions.size(); fun++) {
            PetscArrayzero(flux, totDim) >> checkError;
            const auto& rhsFluxFunctionDescription = discontinuousFluxFunctionDescriptions[fun];
            rhsFluxFunctionDescription.function(
                dim, fg, &uOff[fun][0], &uOff_x[fun][0], uL, uR, gradL, gradR, &aOff[fun][0], &aOff_x[fun][0], auxL, auxR, gradAuxL, gradAuxR, flux, rhsFluxFunctionDescription.context) >>
                checkError;

            // add the flux back to the cell
            PetscScalar *fL = nullptr, *fR = nullptr;
            PetscInt cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[0], fluxId[fun], locFArray, &fL) >> checkError;
            }

            cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[1], fluxId[fun], locFArray, &fR) >> checkError;
            }

            for (PetscInt d = 0; d < fluxComponentSize[fun]; ++d) {
                if (fL) fL[d] -= flux[d] / cgL->volume;
                if (fR) fR[d] += flux[d] / cgR->volume;
            }
        }
    }

    // cleanup
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &flux) >> checkError;
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &uL) >> checkError;
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &uR) >> checkError;
    DMRestoreWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradL) >> checkError;
    DMRestoreWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradR) >> checkError;

    if (dmAux) {
        DMRestoreWorkArray(dmAux, totDimAux, MPIU_SCALAR, &auxL) >> checkError;
        DMRestoreWorkArray(dmAux, totDimAux, MPIU_SCALAR, &auxR) >> checkError;

        DMRestoreWorkArray(dmAux, dim * totDimAux, MPIU_SCALAR, &gradAuxR) >> checkError;
        DMRestoreWorkArray(dmAux, dim * totDimAux, MPIU_SCALAR, &gradAuxL) >> checkError;
    }

    RestoreRange(faceIS, fStart, fEnd, faces);
}
void ablate::finiteVolume::FiniteVolumeSolver::ComputePointSourceTerms(DM dm, PetscDS ds, PetscInt totDim, PetscReal time, const PetscScalar* xArray, DM dmAux, PetscDS dsAux, PetscInt totDimAux,
                                                                       const PetscScalar* auxArray, DM faceDM, const PetscScalar* faceGeomArray, DM cellDM, const PetscScalar* cellGeomArray,
                                                                       std::vector<DM>& dmGrads, std::vector<const PetscScalar*>& locGradArrays, std::vector<DM>& dmAuxGrads,
                                                                       std::vector<const PetscScalar*>& locAuxGradArrays, PetscScalar* locFArray) {
    // Precompute the offsets to pass into the rhsFluxFunctionDescriptions
    std::vector<std::vector<PetscInt>> fluxComponentSize(pointFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> fluxComponentOffset(pointFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> uOff(pointFunctionDescriptions.size());
    std::vector<std::vector<PetscInt>> aOff(pointFunctionDescriptions.size());

    // Get the full set of offsets from the ds
    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(ds, &uOffTotal) >> checkError;

    for (std::size_t fun = 0; fun < pointFunctionDescriptions.size(); fun++) {
        for (std::size_t f = 0; f < pointFunctionDescriptions[fun].fields.size(); f++) {
            const auto& field = subDomain->GetField(pointFunctionDescriptions[fun].fields[f]);

            PetscInt fieldSize, fieldOffset;
            PetscDSGetFieldSize(ds, field.subId, &fieldSize) >> checkError;
            PetscDSGetFieldOffset(ds, field.subId, &fieldOffset) >> checkError;
            fluxComponentSize[fun].push_back(fieldSize);
            fluxComponentOffset[fun].push_back(fieldOffset);
        }

        for (std::size_t f = 0; f < pointFunctionDescriptions[fun].inputFields.size(); f++) {
            uOff[fun].push_back(uOffTotal[pointFunctionDescriptions[fun].inputFields[f]]);
        }
    }

    if (dsAux) {
        PetscInt* auxOffTotal;
        PetscDSGetComponentOffsets(dsAux, &auxOffTotal) >> checkError;
        for (std::size_t fun = 0; fun < pointFunctionDescriptions.size(); fun++) {
            for (std::size_t f = 0; f < pointFunctionDescriptions[fun].auxFields.size(); f++) {
                aOff[fun].push_back(auxOffTotal[pointFunctionDescriptions[fun].auxFields[f]]);
            }
        }
    }

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

    PetscInt dim = subDomain->GetDimensions();

    // Size up a scratch variable
    PetscScalar* fScratch;
    PetscCalloc1(totDim, &fScratch);

    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt* cells;
    GetCellRange(cellIS, cStart, cEnd, cells);

    // Get the pointer to the local grad arrays
    const PetscScalar* const* gradU = locGradArrays.empty() ? nullptr : &locGradArrays[0];
    const PetscScalar* const* gradAux = locAuxGradArrays.empty() ? nullptr : &locAuxGradArrays[0];

    // March over each cell
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // make sure that this is not a ghost cell
        if (ghostLabel) {
            PetscInt ghostVal;

            DMLabelGetValue(ghostLabel, cell, &ghostVal) >> checkError;
            if (ghostVal > 0) continue;
        }

        // extract the point locations for this cell
        const PetscFVCellGeom* cg;
        const PetscScalar* u;
        PetscScalar* rhs;
        DMPlexPointLocalRead(cellDM, cell, cellGeomArray, &cg) >> checkError;
        DMPlexPointLocalRead(dm, cell, xArray, &u) >> checkError;
        DMPlexPointLocalRef(dm, cell, locFArray, &rhs) >> checkError;

        // if there is an aux field, get it
        const PetscScalar* a = nullptr;
        if (auxArray) {
            DMPlexPointLocalRead(dmAux, cell, auxArray, &a) >> checkError;
        }

        // March over each functionDescriptions
        for (std::size_t fun = 0; fun < pointFunctionDescriptions.size(); fun++) {
            // (PetscInt dim, const PetscFVCellGeom *cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void *ctx)
            pointFunctionDescriptions[fun].function(dim, time, cg, &uOff[fun][0], u, gradU, &aOff[fun][0], a, gradAux, fScratch, pointFunctionDescriptions[fun].context) >> checkError;

            // copy over each result flux field
            PetscInt r = 0;
            for (std::size_t ff = 0; ff < pointFunctionDescriptions[fun].fields.size(); ff++) {
                for (PetscInt d = 0; d < fluxComponentSize[fun][ff]; ++d) {
                    rhs[fluxComponentOffset[fun][ff] + d] += fScratch[r++];
                }
            }
        }
    }

    RestoreRange(cellIS, cStart, cEnd, cells);
}
std::map<std::string, double> ablate::finiteVolume::FiniteVolumeSolver::ComputePhysicsTimeSteps(TS ts) {
    // time steps
    std::map<std::string, double> timeSteps;

    // march over each calculator
    for (const auto& dtFunction : timeStepFunctions) {
        double dt = dtFunction.function(ts, *this, dtFunction.context);
        PetscReal dtMinGlobal;
        MPI_Reduce(&dt, &dtMinGlobal, 1, MPIU_REAL, MPI_MIN, 0, PetscObjectComm((PetscObject)ts)) >> checkMpiError;
        timeSteps[dtFunction.name] = dt;
    }

    return timeSteps;
}
bool ablate::finiteVolume::FiniteVolumeSolver::Serialize() const {
    return std::count_if(processes.begin(), processes.end(), [](auto& testProcess) {
        auto serializable = std::dynamic_pointer_cast<ablate::io::Serializable>(testProcess);
        return serializable != nullptr && serializable->Serialize();
    });
}

void ablate::finiteVolume::FiniteVolumeSolver::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    for (auto& process : processes) {
        if (auto serializablePtr = std::dynamic_pointer_cast<ablate::io::Serializable>(process)) {
            if (serializablePtr->Serialize()) {
                serializablePtr->Save(viewer, sequenceNumber, time);
            }
        }
    }
}

void ablate::finiteVolume::FiniteVolumeSolver::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    for (auto& process : processes) {
        if (auto serializablePtr = std::dynamic_pointer_cast<ablate::io::Serializable>(process)) {
            if (serializablePtr->Serialize()) {
                serializablePtr->Restore(viewer, sequenceNumber, time);
            }
        }
    }
}

/**
 * Hard coded function to compute the boundary cell gradient.  This should be relaxed to a boundary condition
 * @param dim
 * @param dof
 * @param faceGeom
 * @param cellGeom
 * @param cellGeomG
 * @param a_xI
 * @param a_xGradI
 * @param a_xG
 * @param a_xGradG
 * @param ctx
 * @return
 */
static PetscErrorCode ComputeBoundaryCellGradient(PetscInt dim, PetscInt dof, const PetscFVFaceGeom* faceGeom, const PetscFVCellGeom* cellGeom, const PetscFVCellGeom* cellGeomG,
                                                  const PetscScalar* a_xI, const PetscScalar* a_xGradI, const PetscScalar* a_xG, PetscScalar* a_xGradG, void* ctx) {
    PetscFunctionBeginUser;

    for (PetscInt pd = 0; pd < dof; ++pd) {
        PetscReal dPhidS = a_xG[pd] - a_xI[pd];

        // over each direction
        for (PetscInt dir = 0; dir < dim; dir++) {
            PetscReal dx = (cellGeomG->centroid[dir] - cellGeom->centroid[dir]) / 2.0;

            // If there is a contribution in this direction
            if (PetscAbs(dx) > 1E-8) {
                a_xGradG[pd * dim + dir] = dPhidS / dx;
            } else {
                a_xGradG[pd * dim + dir] = 0.0;
            }
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::FillGradientBoundary(DM dm, PetscFV auxFvm, Vec localXVec, Vec gradLocalVec) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Get the dmGrad
    DM dmGrad;
    ierr = VecGetDM(gradLocalVec, &dmGrad);
    CHKERRQ(ierr);

    // get the problem
    PetscDS prob;
    ierr = DMGetDS(dm, &prob);
    CHKERRQ(ierr);

    PetscInt field;
    ierr = PetscDSGetFieldIndex(prob, (PetscObject)auxFvm, &field);
    CHKERRQ(ierr);
    PetscInt dof;
    ierr = PetscDSGetFieldSize(prob, field, &dof);
    CHKERRQ(ierr);

    // Obtaining local cell ownership
    PetscInt faceStart, faceEnd;
    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);
    CHKERRQ(ierr);

    // Get the fvm face and cell geometry
    CHKERRQ(ierr);

    // get the dm for each geom type
    DM dmFaceGeom, dmCellGeom;
    ierr = VecGetDM(faceGeomVec, &dmFaceGeom);
    CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCellGeom);
    CHKERRQ(ierr);

    // extract the gradLocalVec
    PetscScalar* gradLocalArray;
    ierr = VecGetArray(gradLocalVec, &gradLocalArray);
    CHKERRQ(ierr);

    const PetscScalar* localArray;
    ierr = VecGetArrayRead(localXVec, &localArray);
    CHKERRQ(ierr);

    // get the dim
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);

    // extract the arrays for the face and cell geom, along with their dm
    const PetscScalar* cellGeomArray;
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);
    CHKERRQ(ierr);

    const PetscScalar* faceGeomArray;
    ierr = VecGetArrayRead(faceGeomVec, &faceGeomArray);
    CHKERRQ(ierr);

    // march over each boundary for this problem
    PetscInt numBd;
    ierr = PetscDSGetNumBoundary(prob, &numBd);
    CHKERRQ(ierr);
    for (PetscInt b = 0; b < numBd; ++b) {
        // extract the boundary information
        PetscInt numids;
        const PetscInt* ids;
        DMLabel label;
        const char* name;
        PetscInt boundaryField;
        ierr = PetscDSGetBoundary(prob, b, nullptr, nullptr, &name, &label, &numids, &ids, &boundaryField, nullptr, nullptr, nullptr, nullptr, nullptr);
        CHKERRQ(ierr);

        if (boundaryField != field) {
            continue;
        }

        // get the correct boundary/ghost pattern
        PetscSF sf;
        PetscInt nleaves;
        const PetscInt* leaves;
        ierr = DMGetPointSF(dm, &sf);
        CHKERRQ(ierr);
        ierr = PetscSFGetGraph(sf, nullptr, &nleaves, &leaves, nullptr);
        CHKERRQ(ierr);
        nleaves = PetscMax(0, nleaves);

        // march over each id on this process
        for (PetscInt i = 0; i < numids; ++i) {
            IS faceIS;
            const PetscInt* faces;
            PetscInt numFaces, f;

            ierr = DMLabelGetStratumIS(label, ids[i], &faceIS);
            CHKERRQ(ierr);
            if (!faceIS) continue; /* No points with that id on this process */
            ierr = ISGetLocalSize(faceIS, &numFaces);
            CHKERRQ(ierr);
            ierr = ISGetIndices(faceIS, &faces);
            CHKERRQ(ierr);

            // march over each face in this boundary
            for (f = 0; f < numFaces; ++f) {
                const PetscInt* cells;

                if ((faces[f] < faceStart) || (faces[f] >= faceEnd)) {
                    continue; /* Refinement adds non-faces to labels */
                }
                PetscInt loc;
                ierr = PetscFindInt(faces[f], nleaves, (PetscInt*)leaves, &loc);
                CHKERRQ(ierr);
                if (loc >= 0) {
                    continue;
                }

                PetscBool boundary;
                ierr = DMIsBoundaryPoint(dm, faces[f], &boundary);
                CHKERRQ(ierr);

                // get the ghost and interior nodes
                ierr = DMPlexGetSupport(dm, faces[f], &cells);
                CHKERRQ(ierr);
                const PetscInt cellI = cells[0];
                const PetscInt cellG = cells[1];

                // get the face geom
                const PetscFVFaceGeom* faceGeom;
                ierr = DMPlexPointLocalRead(dmFaceGeom, faces[f], faceGeomArray, &faceGeom);
                CHKERRQ(ierr);

                // get the cell centroid information
                const PetscFVCellGeom* cellGeom;
                const PetscFVCellGeom* cellGeomGhost;
                ierr = DMPlexPointLocalRead(dmCellGeom, cellI, cellGeomArray, &cellGeom);
                CHKERRQ(ierr);
                ierr = DMPlexPointLocalRead(dmCellGeom, cellG, cellGeomArray, &cellGeomGhost);
                CHKERRQ(ierr);

                // Read the local point
                PetscScalar* boundaryGradCellValues;
                ierr = DMPlexPointLocalRef(dmGrad, cellG, gradLocalArray, &boundaryGradCellValues);
                CHKERRQ(ierr);

                const PetscScalar* cellGradValues;
                ierr = DMPlexPointLocalRead(dmGrad, cellI, gradLocalArray, &cellGradValues);
                CHKERRQ(ierr);

                const PetscScalar* boundaryCellValues;
                ierr = DMPlexPointLocalFieldRead(dm, cellG, field, localArray, &boundaryCellValues);
                CHKERRQ(ierr);
                const PetscScalar* cellValues;
                ierr = DMPlexPointLocalFieldRead(dm, cellI, field, localArray, &cellValues);
                CHKERRQ(ierr);

                // compute the gradient for the boundary node and pass in
                ierr = ComputeBoundaryCellGradient(dim, dof, faceGeom, cellGeom, cellGeomGhost, cellValues, cellGradValues, boundaryCellValues, boundaryGradCellValues, nullptr);
                CHKERRQ(ierr);
            }
            ierr = ISRestoreIndices(faceIS, &faces);
            CHKERRQ(ierr);
            ierr = ISDestroy(&faceIS);
            CHKERRQ(ierr);
        }
    }

    ierr = VecRestoreArrayRead(localXVec, &localArray);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(gradLocalVec, &gradLocalArray);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(faceGeomVec, &faceGeomArray);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode BuildGradientReconstruction_Internal(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, DM dmFace, PetscScalar* fgeom, DM dmCell, PetscScalar* cgeom) {
    DMLabel ghostLabel;
    PetscScalar *dx, *grad, **gref;
    PetscInt dim, cStart, cEnd, c, cEndInterior, maxNumFaces;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
    cEndInterior = cEndInterior < 0 ? cEnd : cEndInterior;
    PetscCall(DMPlexGetMaxSizes(dm, &maxNumFaces, NULL));
    PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
    PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
    PetscCall(PetscMalloc3(maxNumFaces * dim, &dx, maxNumFaces * dim, &grad, maxNumFaces, &gref));
    for (c = cStart; c < cEndInterior; c++) {
        const PetscInt* faces;
        PetscInt numFaces, usedFaces, f, d;
        PetscFVCellGeom* cg;
        PetscBool boundary;
        PetscInt ghost;
        PetscInt labelValue;

        // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
        PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
        if (ghost >= 0) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
        PetscCall(DMPlexGetConeSize(dm, c, &numFaces));
        PetscCall(DMPlexGetCone(dm, c, &faces));
        PetscCheckFalse(numFaces < dim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %D has only %D faces, not enough for gradient reconstruction", c, numFaces);
        for (f = 0, usedFaces = 0; f < numFaces; ++f) {
            PetscFVCellGeom* cg1;
            PetscFVFaceGeom* fg;
            const PetscInt* fcells;
            PetscInt ncell, side;

            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, faces[f], &labelValue));
                if (labelValue != regionValue) continue;
            }

            PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
            PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
            if ((ghost >= 0) || boundary) continue;
            PetscCall(DMPlexGetSupport(dm, faces[f], &fcells));
            side = (c != fcells[0]); /* c is on left=0 or right=1 of face */
            ncell = fcells[!side];   /* the neighbor */
            PetscCall(DMPlexPointLocalRef(dmFace, faces[f], fgeom, &fg));
            PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
            for (d = 0; d < dim; ++d) dx[usedFaces * dim + d] = cg1->centroid[d] - cg->centroid[d];
            gref[usedFaces++] = fg->grad[side]; /* Gradient reconstruction term will go here */
        }
        PetscCheck(usedFaces, PETSC_COMM_SELF, PETSC_ERR_USER, "Mesh contains isolated cell (no neighbors). Is it intentional?");
        PetscCall(PetscFVComputeGradient(fvm, usedFaces, dx, grad));
        for (f = 0, usedFaces = 0; f < numFaces; ++f) {
            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, faces[f], &labelValue));
                if (labelValue != regionValue) continue;
            }
            PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
            PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
            if ((ghost >= 0) || boundary) continue;
            for (d = 0; d < dim; ++d) gref[usedFaces][d] = grad[usedFaces * dim + d];
            ++usedFaces;
        }
    }
    PetscCall(PetscFree3(dx, grad, gref));
    PetscFunctionReturn(0);
}

static PetscErrorCode BuildGradientReconstruction_Internal_Tree(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, DM dmFace, PetscScalar* fgeom, DM dmCell, PetscScalar* cgeom) {
    DMLabel ghostLabel;
    PetscScalar *dx, *grad, **gref;
    PetscInt dim, cStart, cEnd, c, cEndInterior, fStart, fEnd, f, nStart, nEnd, maxNumFaces = 0;
    PetscSection neighSec;
    PetscInt(*neighbors)[2];
    PetscInt* counter;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
    if (cEndInterior < 0) cEndInterior = cEnd;
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &neighSec));
    PetscCall(PetscSectionSetChart(neighSec, cStart, cEndInterior));
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
    for (f = fStart; f < fEnd; f++) {
        const PetscInt* fcells;
        PetscBool boundary;
        PetscInt ghost = -1;
        PetscInt numChildren, numCells, labelValue;

        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
        PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
        PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, NULL));
        if ((ghost >= 0) || boundary || numChildren) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
        if (numCells == 2) {
            PetscCall(DMPlexGetSupport(dm, f, &fcells));
            for (c = 0; c < 2; c++) {
                PetscInt cell = fcells[c];

                if (cell >= cStart && cell < cEndInterior) {
                    PetscCall(PetscSectionAddDof(neighSec, cell, 1));
                }
            }
        }
    }
    PetscCall(PetscSectionSetUp(neighSec));
    PetscCall(PetscSectionGetMaxDof(neighSec, &maxNumFaces));
    PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
    nStart = 0;
    PetscCall(PetscSectionGetStorageSize(neighSec, &nEnd));
    PetscCall(PetscMalloc1((nEnd - nStart), &neighbors));
    PetscCall(PetscCalloc1((cEndInterior - cStart), &counter));
    for (f = fStart; f < fEnd; f++) {
        const PetscInt* fcells;
        PetscBool boundary;
        PetscInt ghost = -1;
        PetscInt numChildren, numCells, labelValue;

        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
        PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
        PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, NULL));
        if ((ghost >= 0) || boundary || numChildren) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
        if (numCells == 2) {
            PetscCall(DMPlexGetSupport(dm, f, &fcells));
            for (c = 0; c < 2; c++) {
                PetscInt cell = fcells[c], off;

                if (regionLabel) {
                    PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
                    if (labelValue != regionValue) continue;
                }

                if (cell >= cStart && cell < cEndInterior) {
                    PetscCall(PetscSectionGetOffset(neighSec, cell, &off));
                    off += counter[cell - cStart]++;
                    neighbors[off][0] = f;
                    neighbors[off][1] = fcells[1 - c];
                }
            }
        }
    }
    PetscCall(PetscFree(counter));
    PetscCall(PetscMalloc3(maxNumFaces * dim, &dx, maxNumFaces * dim, &grad, maxNumFaces, &gref));
    for (c = cStart; c < cEndInterior; c++) {
        PetscInt numFaces, d, off, labelValue, ghost = -1;
        PetscFVCellGeom* cg;

        PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
        PetscCall(PetscSectionGetDof(neighSec, c, &numFaces));
        PetscCall(PetscSectionGetOffset(neighSec, c, &off));

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
            if (labelValue != regionValue) continue;
        }

        // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
        if (ghost >= 0) continue;

        PetscCheckFalse(numFaces < dim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %D has only %D faces, not enough for gradient reconstruction", c, numFaces);
        for (f = 0; f < numFaces; ++f) {
            PetscFVCellGeom* cg1;
            PetscFVFaceGeom* fg;
            const PetscInt* fcells;
            PetscInt ncell, side, nface;

            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
                if (labelValue != regionValue) continue;
            }

            nface = neighbors[off + f][0];
            ncell = neighbors[off + f][1];
            PetscCall(DMPlexGetSupport(dm, nface, &fcells));
            side = (c != fcells[0]);
            PetscCall(DMPlexPointLocalRef(dmFace, nface, fgeom, &fg));
            PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
            for (d = 0; d < dim; ++d) dx[f * dim + d] = cg1->centroid[d] - cg->centroid[d];
            gref[f] = fg->grad[side]; /* Gradient reconstruction term will go here */
        }
        PetscCall(PetscFVComputeGradient(fvm, numFaces, dx, grad));
        for (f = 0; f < numFaces; ++f) {
            for (d = 0; d < dim; ++d) gref[f][d] = grad[f * dim + d];
        }
    }
    PetscCall(PetscFree3(dx, grad, gref));
    PetscCall(PetscSectionDestroy(&neighSec));
    PetscCall(PetscFree(neighbors));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::ComputeGradientFVM(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, Vec faceGeometry, Vec cellGeometry, DM* dmGrad) {
    DM dmFace, dmCell;
    PetscScalar *fgeom, *cgeom;
    PetscSection sectionGrad, parentSection;
    PetscInt dim, pdim, cStart, cEnd, cEndInterior, c;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscFVGetNumComponents(fvm, &pdim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, NULL));
    /* Construct the interpolant corresponding to each face from the least-square solution over the cell neighborhood */
    PetscCall(VecGetDM(faceGeometry, &dmFace));
    PetscCall(VecGetDM(cellGeometry, &dmCell));
    PetscCall(VecGetArray(faceGeometry, &fgeom));
    PetscCall(VecGetArray(cellGeometry, &cgeom));
    PetscCall(DMPlexGetTree(dm, &parentSection, NULL, NULL, NULL, NULL));
    if (!parentSection) {
        PetscCall(BuildGradientReconstruction_Internal(dm, regionLabel, regionValue, fvm, dmFace, fgeom, dmCell, cgeom));
    } else {
        PetscCall(BuildGradientReconstruction_Internal_Tree(dm, regionLabel, regionValue, fvm, dmFace, fgeom, dmCell, cgeom));
    }
    PetscCall(VecRestoreArray(faceGeometry, &fgeom));
    PetscCall(VecRestoreArray(cellGeometry, &cgeom));
    /* Create storage for gradients */
    PetscCall(DMClone(dm, dmGrad));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionGrad));
    PetscCall(PetscSectionSetChart(sectionGrad, cStart, cEnd));
    for (c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(sectionGrad, c, pdim * dim));
    PetscCall(PetscSectionSetUp(sectionGrad));
    PetscCall(DMSetLocalSection(*dmGrad, sectionGrad));
    PetscCall(PetscSectionDestroy(&sectionGrad));
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::FiniteVolumeSolver, "finite volume solver", ARG(std::string, "id", "the name of the flow field"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(std::vector<ablate::finiteVolume::processes::Process>, "processes", "the processes used to describe the flow"),
         OPT(std::vector<ablate::finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(bool, "computePhysicsTimeStep", "determines if a physics based time step is used to control the FVM time stepping (default is false)"));

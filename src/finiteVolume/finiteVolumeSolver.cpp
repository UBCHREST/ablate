#include "finiteVolumeSolver.hpp"
#include <utility>
#include "cellInterpolant.hpp"
#include "faceInterpolant.hpp"
#include "processes/process.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

ablate::finiteVolume::FiniteVolumeSolver::FiniteVolumeSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                             std::vector<std::shared_ptr<processes::Process>> processes,
                                                             std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions)
    : CellSolver(std::move(solverId), std::move(region), std::move(options)),
      processes(std::move(processes)),
      boundaryConditions(std::move(boundaryConditions)),
      solverRegionMinusGhost(std::make_shared<domain::Region>(solverId + "_minusGhost")) {}

ablate::finiteVolume::FiniteVolumeSolver::~FiniteVolumeSolver() {
    if (meshCharacteristicsLocalVec) {
        VecDestroy(&meshCharacteristicsLocalVec) >> utilities::PetscUtilities::checkError;
    }
    if (meshCharacteristicsDm) {
        DMDestroy(&meshCharacteristicsDm) >> utilities::PetscUtilities::checkError;
    }
}

void ablate::finiteVolume::FiniteVolumeSolver::Setup() {
    ablate::solver::CellSolver::Setup();

    // march over process and link to the flow
    for (const auto& process : processes) {
        process->Setup(*this);
    }

    // Set the flux calculator solver for each component
    PetscDSSetFromOptions(subDomain->GetDiscreteSystem()) >> utilities::PetscUtilities::checkError;

    // Some petsc code assumes that a ghostLabel has created, so create one
    PetscBool ghostLabel;
    DMHasLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;
    if (!ghostLabel) {
        throw std::runtime_error("The FiniteVolumeSolver expects ghost cells around the boundary even if the FiniteVolumeSolver region does not include the boundary.");
    }
}

void ablate::finiteVolume::FiniteVolumeSolver::Initialize() {
    // call the base class Initialize
    ablate::solver::CellSolver::Initialize();

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
        PetscDSGetNumBoundary(flowProblem, &numberBC) >> utilities::PetscUtilities::checkError;
        PetscInt numberAuxFields;
        PetscDSGetNumFields(auxProblem, &numberAuxFields) >> utilities::PetscUtilities::checkError;

        for (PetscInt bc = 0; bc < numberBC; bc++) {
            DMBoundaryConditionType type;
            const char* name;
            DMLabel label;
            PetscInt field;
            PetscInt numberIds;
            const PetscInt* ids;

            // Get the boundary
            PetscDSGetBoundary(flowProblem, bc, nullptr, &type, &name, &label, &numberIds, &ids, &field, nullptr, nullptr, nullptr, nullptr, nullptr) >> utilities::PetscUtilities::checkError;

            // If this is for euler and DM_BC_NATURAL_RIEMANN add it to the aux
            if (type == DM_BC_NATURAL_RIEMANN && field == 0) {
                for (PetscInt af = 0; af < numberAuxFields; af++) {
                    PetscDSAddBoundary(auxProblem, type, name, label, numberIds, ids, af, 0, nullptr, nullptr, nullptr, nullptr, nullptr) >> utilities::PetscUtilities::checkError;
                }
            }
        }
    }

    {  // get the cell is for the solver minus ghost cell
        // Get the original range
        ablate::domain::Range cellRange;
        GetCellRange(cellRange);

        // create a new label
        auto dm = GetSubDomain().GetDM();
        DMCreateLabel(dm, solverRegionMinusGhost->GetName().c_str()) >> utilities::PetscUtilities::checkError;
        DMLabel solverRegionMinusGhostLabel;
        PetscInt solverRegionMinusGhostValue;
        domain::Region::GetLabel(solverRegionMinusGhost, dm, solverRegionMinusGhostLabel, solverRegionMinusGhostValue);

        // Get the ghost cell label
        DMLabel ghostLabel;
        DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

        // check if it is an exterior boundary cell ghost
        PetscInt boundaryCellStart;
        DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &boundaryCellStart, nullptr) >> utilities::PetscUtilities::checkError;

        // march over every cell
        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            PetscInt cell = cellRange.points ? cellRange.points[c] : c;

            // check if it is boundary ghost
            PetscInt isGhost = -1;
            if (ghostLabel) {
                DMLabelGetValue(ghostLabel, cell, &isGhost) >> utilities::PetscUtilities::checkError;
            }

            PetscInt owned;
            DMPlexGetPointGlobal(dm, cell, &owned, nullptr) >> utilities::PetscUtilities::checkError;
            if (owned >= 0 && isGhost < 0 && (boundaryCellStart < 0 || cell < boundaryCellStart)) {
                DMLabelSetValue(solverRegionMinusGhostLabel, cell, solverRegionMinusGhostValue);
            }
        }
        RestoreRange(cellRange);
    }

    // march over process and link to the new mesh
    for (const auto& process : processes) {
        process->Initialize(*this);
    }

    // Create the field to hold the max/max
    DMClone(subDomain->GetDM(), &meshCharacteristicsDm);
    PetscBool simplex;
    DMPlexIsSimplex(subDomain->GetDM(), &simplex) >> utilities::PetscUtilities::checkError;
    {  // Create the minCellRadius field
        PetscFE field;
        PetscFECreateLagrange(PETSC_COMM_SELF, subDomain->GetDimensions(), 1, simplex, 0 /**cell center**/, PETSC_DETERMINE, &field) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)field, "minCellRadius") >> utilities::PetscUtilities::checkError;
        DMSetField(meshCharacteristicsDm, MIN_CELL_RADIUS, nullptr, (PetscObject)field) >> utilities::PetscUtilities::checkError;
        PetscFEDestroy(&field) >> utilities::PetscUtilities::checkError;
    }
    {  // Create the maxCellRadius field
        PetscFE field;
        PetscFECreateLagrange(PETSC_COMM_SELF, subDomain->GetDimensions(), 1, simplex, 0 /**cell center**/, PETSC_DETERMINE, &field) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)field, "maxCellRadius") >> utilities::PetscUtilities::checkError;
        DMSetField(meshCharacteristicsDm, MAX_CELL_RADIUS, nullptr, (PetscObject)field) >> utilities::PetscUtilities::checkError;
        PetscFEDestroy(&field) >> utilities::PetscUtilities::checkError;
    }

    // add the field
    DMCreateDS(meshCharacteristicsDm) >> utilities::PetscUtilities::checkError;

    // Create a vector to store the result
    DMCreateLocalVector(meshCharacteristicsDm, &meshCharacteristicsLocalVec) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)meshCharacteristicsLocalVec, "meshCharacteristics") >> utilities::PetscUtilities::checkError;
    VecSet(meshCharacteristicsLocalVec, NAN) >> utilities::PetscUtilities::checkError;

    PetscScalar* meshCharacteristicsLocalArray;
    VecGetArray(meshCharacteristicsLocalVec, &meshCharacteristicsLocalArray) >> utilities::PetscUtilities::checkError;

    // get the face/cell information
    DM faceDM, cellDM;
    VecGetDM(faceGeomVec, &faceDM) >> utilities::PetscUtilities::checkError;
    VecGetDM(cellGeomVec, &cellDM) >> utilities::PetscUtilities::checkError;
    const PetscScalar* cellGeomArray;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

    DMLabel ghostLabel = nullptr;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(subDomain->GetDM(), 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;

    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscInt face = f;

        // make sure that this is a valid face
        PetscInt ghost = -1, ncells, nchild;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
        }
        DMPlexGetSupportSize(subDomain->GetDM(), face, &ncells) >> utilities::PetscUtilities::checkError;
        DMPlexGetTreeChildren(subDomain->GetDM(), face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
        if (ghost >= 0 || !ncells || nchild > 0) continue;

        // Get the face geometry
        const PetscInt* faceCells;
        PetscFVFaceGeom* fg;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupport(subDomain->GetDM(), face, &faceCells) >> utilities::PetscUtilities::checkError;

        // compute first cell
        for (PetscInt c = 0; c < ncells; ++c) {
            PetscScalar* meshCharacteristics;
            PetscFVCellGeom* cg;
            DMPlexPointLocalRef(meshCharacteristicsDm, faceCells[c], meshCharacteristicsLocalArray, &meshCharacteristics) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalRead(cellDM, faceCells[c], cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;
            if (meshCharacteristics) {
                // compute max min
                PetscScalar v[3];
                utilities::MathUtilities::Subtract(subDomain->GetDimensions(), cg->centroid, fg->centroid, v);
                PetscReal radius = utilities::MathUtilities::MagVector(subDomain->GetDimensions(), v);

                if (PetscIsNanScalar(meshCharacteristics[MIN_CELL_RADIUS])) {
                    meshCharacteristics[MIN_CELL_RADIUS] = radius;
                    meshCharacteristics[MAX_CELL_RADIUS] = radius;
                } else {
                    meshCharacteristics[MIN_CELL_RADIUS] = PetscMin(meshCharacteristics[MIN_CELL_RADIUS], radius);
                    meshCharacteristics[MAX_CELL_RADIUS] = PetscMax(meshCharacteristics[MAX_CELL_RADIUS], radius);
                }
            }
        }
    }

    VecRestoreArray(meshCharacteristicsLocalVec, &meshCharacteristicsLocalArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
}

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) {
    PetscFunctionBeginUser;
    ablate::domain::Range faceRange, cellRange;
    GetFaceRange(faceRange);
    GetCellRange(cellRange);
    try {
        StartEvent("FiniteVolumeSolver::ComputeRHSFunction::discontinuousFluxFunction");
        if (!discontinuousFluxFunctionDescriptions.empty()) {
            if (cellInterpolant == nullptr) {
                cellInterpolant = std::make_unique<CellInterpolant>(subDomain, GetRegion(), faceGeomVec, cellGeomVec);
            }

            cellInterpolant->ComputeRHS(time, locXVec, subDomain->GetAuxVector(), locFVec, GetRegion(), discontinuousFluxFunctionDescriptions, faceRange, cellRange, cellGeomVec, faceGeomVec);
        }
        EndEvent();
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in CellInterpolant discontinuousFluxFunction: %s", exception.what());
    }

    try {
        StartEvent("FiniteVolumeSolver::ComputeRHSFunction::pointFunction");
        if (!pointFunctionDescriptions.empty()) {
            if (cellInterpolant == nullptr) {
                cellInterpolant = std::make_unique<CellInterpolant>(subDomain, GetRegion(), faceGeomVec, cellGeomVec);
            }

            cellInterpolant->ComputeRHS(time, locXVec, subDomain->GetAuxVector(), locFVec, GetRegion(), pointFunctionDescriptions, cellRange, cellGeomVec);
        }
        EndEvent();
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in CellInterpolant pointFunctionDescriptions: %s", exception.what());
    }

    try {
        StartEvent("FiniteVolumeSolver::ComputeRHSFunction::continuousFluxFunctionDescriptions");
        if (!continuousFluxFunctionDescriptions.empty()) {
            if (faceInterpolant == nullptr) {
                faceInterpolant = std::make_unique<FaceInterpolant>(subDomain, GetRegion(), faceGeomVec, cellGeomVec);
            }

            faceInterpolant->ComputeRHS(time, locXVec, subDomain->GetAuxVector(), locFVec, GetRegion(), continuousFluxFunctionDescriptions, faceRange, cellGeomVec, faceGeomVec);
        }
        EndEvent();
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in FaceInterpolant continuousFluxFunctionDescriptions: %s", exception.what());
    }

    RestoreRange(faceRange);
    RestoreRange(cellRange);

    // iterate over any arbitrary RHS functions
    StartEvent("FiniteVolumeSolver::ComputeRHSFunction::rhsArbitraryFunctions");
    for (const auto& rhsFunction : rhsArbitraryFunctions) {
        PetscCall(rhsFunction.first(*this, subDomain->GetDM(), time, locXVec, locFVec, rhsFunction.second));
    }
    EndEvent();

    PetscFunctionReturn(0);
}

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(CellInterpolant::DiscontinuousFluxFunction function, void* context, const std::string& field,
                                                                   const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields) {
    // map the field, inputFields, and auxFields to locations
    auto& fieldId = subDomain->GetField(field);

    // Create the FVMRHS Function
    CellInterpolant::DiscontinuousFluxFunctionDescription functionDescription{.function = function, .context = context, .field = fieldId.id};

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

void ablate::finiteVolume::FiniteVolumeSolver::RegisterRHSFunction(CellInterpolant::PointFunction function, void* context, const std::vector<std::string>& fields,
                                                                   const std::vector<std::string>& inputFields, const std::vector<std::string>& auxFields) {
    // Create the FVMRHS Function
    CellInterpolant::PointFunctionDescription functionDescription{.function = function, .context = context};

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

void ablate::finiteVolume::FiniteVolumeSolver::RegisterPreRHSFunction(PreRHSFunctionDefinition function, void* context) { preRhsFunctions.emplace_back(function, context); }

void ablate::finiteVolume::FiniteVolumeSolver::RegisterComputeTimeStepFunction(ComputeTimeStepFunction function, void* ctx, std::string name) {
    timeStepFunctions.emplace_back(ComputeTimeStepDescription{.function = function, .context = ctx, .name = std::move(name)});
}

double ablate::finiteVolume::FiniteVolumeSolver::ComputePhysicsTimeStep(TS ts) {
    // march over each calculator
    PetscReal dtMin = ablate::utilities::Constants::large;
    for (const auto& dtFunction : timeStepFunctions) {
        dtMin = PetscMin(dtMin, dtFunction.function(ts, *this, dtFunction.context));
    }

    return dtMin;
}

std::map<std::string, double> ablate::finiteVolume::FiniteVolumeSolver::ComputePhysicsTimeSteps(TS ts) {
    // time steps
    std::map<std::string, double> timeSteps;

    // march over each calculator
    for (const auto& dtFunction : timeStepFunctions) {
        double dt = dtFunction.function(ts, *this, dtFunction.context);
        PetscReal dtMinGlobal;
        MPI_Reduce(&dt, &dtMinGlobal, 1, MPIU_REAL, MPI_MIN, 0, PetscObjectComm((PetscObject)ts)) >> ablate::utilities::MpiUtilities::checkError;
        timeSteps[dtFunction.name] = dtMinGlobal;
    }

    return timeSteps;
}

ablate::io::Serializable::SerializerType ablate::finiteVolume::FiniteVolumeSolver::Serialize() const {
    return DetermineSerializerType(processes);

}

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    for (auto& process : processes) {
        if (auto serializablePtr = std::dynamic_pointer_cast<ablate::io::Serializable>(process)) {
            if (serializablePtr->Serialize() != SerializerType::none) {
                PetscCall(serializablePtr->Save(viewer, sequenceNumber, time));
            }
        }
    }

    // On the mesh output also save the mesh information
    if (sequenceNumber == 0 && time == 0.0) {
        // We need to save this as global vector
        Vec meshCharacteristicsGlobVec;
        PetscCall(DMGetGlobalVector(meshCharacteristicsDm, &meshCharacteristicsGlobVec));

        // Copy the aux vector name
        const char* vecName;
        PetscCall(PetscObjectGetName((PetscObject)meshCharacteristicsLocalVec, &vecName));
        PetscCall(PetscObjectSetName((PetscObject)meshCharacteristicsGlobVec, vecName));

        // copy from local to global
        PetscCall(DMLocalToGlobal(meshCharacteristicsDm, meshCharacteristicsLocalVec, INSERT_VALUES, meshCharacteristicsGlobVec));
        PetscCall(DMSetOutputSequenceNumber(meshCharacteristicsDm, sequenceNumber, time));
        PetscCall(DMView(meshCharacteristicsDm, viewer));
        PetscCall(VecView(meshCharacteristicsGlobVec, viewer));

        // clean up
        PetscCall(DMRestoreGlobalVector(meshCharacteristicsDm, &meshCharacteristicsGlobVec));
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    for (auto& process : processes) {
        if (auto serializablePtr = std::dynamic_pointer_cast<ablate::io::Serializable>(process)) {
            if (serializablePtr->Serialize() != SerializerType::none) {
                PetscCall(serializablePtr->Restore(viewer, sequenceNumber, time));
            }
        }
    }
    PetscFunctionReturn(0);
}

void ablate::finiteVolume::FiniteVolumeSolver::GetCellRangeWithoutGhost(ablate::domain::Range& faceRange) const {
    // Get the point range
    DMLabel solverRegionMinusGhostLabel;
    PetscInt solverRegionMinusGhostValue;
    domain::Region::GetLabel(solverRegionMinusGhost, GetSubDomain().GetDM(), solverRegionMinusGhostLabel, solverRegionMinusGhostValue);

    DMLabelGetStratumIS(solverRegionMinusGhostLabel, solverRegionMinusGhostValue, &faceRange.is) >> utilities::PetscUtilities::checkError;
    if (faceRange.is == nullptr) {
        // There are no points in this region, so skip
        faceRange.start = 0;
        faceRange.end = 0;
        faceRange.points = nullptr;
    } else {
        // Get the range
        ISGetPointRange(faceRange.is, &faceRange.start, &faceRange.end, &faceRange.points) >> utilities::PetscUtilities::checkError;
    }
}

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::ComputeBoundary(PetscReal time, Vec locX, Vec locX_t) {
    PetscFunctionBeginUser;
    auto dm = subDomain->GetDM();
    auto ds = subDomain->GetDiscreteSystem();
    /* Handle non-essential (e.g. outflow) boundary values.  This should be done before the auxFields are updated so that boundary values can be updated */
    PetscCall(ablate::solver::Solver::DMPlexInsertBoundaryValues_Plex(dm, ds, PETSC_FALSE, locX, time, faceGeomVec, cellGeomVec, nullptr));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::PreRHSFunction(TS ts, PetscReal time, bool initialStage, Vec locX) {
    PetscFunctionBeginUser;
    StartEvent("FiniteVolumeSolver::PreRHSFunction");
    try {
        // update any aux fields, including ghost cells
        UpdateAuxFields(time, locX, subDomain->GetAuxVector());
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in UpdateAuxFields: %s", exception.what());
    }
    // iterate over any pre arbitrary RHS functions
    for (const auto& rhsFunction : preRhsFunctions) {
        PetscCall(rhsFunction.first(*this, ts, time, initialStage, locX, rhsFunction.second));
    }
    EndEvent();
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::FiniteVolumeSolver, "finite volume solver", ARG(std::string, "id", "the name of the flow field"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(std::vector<ablate::finiteVolume::processes::Process>, "processes", "the processes used to describe the flow"),
         OPT(std::vector<ablate::finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"));

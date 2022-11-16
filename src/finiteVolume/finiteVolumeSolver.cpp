#include "finiteVolumeSolver.hpp"
#include <utility>
#include "cellInterpolant.hpp"
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
      boundaryConditions(std::move(boundaryConditions)),
      solverRegionMinusGhost(std::make_shared<domain::Region>(solverId + "_minusGhost")) {}

void ablate::finiteVolume::FiniteVolumeSolver::Setup() {
    ablate::solver::CellSolver::Setup();

    // march over process and link to the flow
    for (const auto& process : processes) {
        process->Setup(*this);
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

    {  // get the cell is for the solver minus ghost cell
        // Get the original range
        solver::Range cellRange;
        GetCellRange(cellRange);

        // create a new label
        auto dm = GetSubDomain().GetDM();
        DMCreateLabel(dm, solverRegionMinusGhost->GetName().c_str()) >> checkError;
        DMLabel solverRegionMinusGhostLabel;
        PetscInt solverRegionMinusGhostValue;
        domain::Region::GetLabel(solverRegionMinusGhost, dm, solverRegionMinusGhostLabel, solverRegionMinusGhostValue);

        // Get the ghost cell label
        DMLabel ghostLabel;
        DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;

        // check if it is an exterior boundary cell ghost
        PetscInt boundaryCellStart;
        DMPlexGetGhostCellStratum(dm, &boundaryCellStart, nullptr) >> checkError;

        // march over every cell
        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            PetscInt cell = cellRange.points ? cellRange.points[c] : c;

            // check if it is boundary ghost
            PetscInt isGhost = -1;
            if (ghostLabel) {
                DMLabelGetValue(ghostLabel, cell, &isGhost) >> checkError;
            }

            PetscInt owned;
            DMPlexGetPointGlobal(dm, cell, &owned, nullptr) >> checkError;
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

    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in UpdateAuxFields: %s", exception.what());
    }

    solver::Range faceRange, cellRange;
    GetFaceRange(faceRange);
    GetCellRange(cellRange);
    try {
        if (!discontinuousFluxFunctionDescriptions.empty()) {
            if (cellInterpolant == nullptr) {
                cellInterpolant = std::make_unique<CellInterpolant>(subDomain, GetRegion(), faceGeomVec, cellGeomVec);
            }

            cellInterpolant->ComputeRHS(time, locXVec, subDomain->GetAuxVector(), locFVec, GetRegion(), discontinuousFluxFunctionDescriptions, faceRange, cellRange, cellGeomVec, faceGeomVec);
        }
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in CellInterpolant discontinuousFluxFunction: %s", exception.what());
    }

    try {
        if (!pointFunctionDescriptions.empty()) {
            if (cellInterpolant == nullptr) {
                cellInterpolant = std::make_unique<CellInterpolant>(subDomain, GetRegion(), faceGeomVec, cellGeomVec);
            }

            cellInterpolant->ComputeRHS(time, locXVec, subDomain->GetAuxVector(), locFVec, GetRegion(), pointFunctionDescriptions, cellRange, cellGeomVec);
        }
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in CellInterpolant pointFunctionDescriptions: %s", exception.what());
    }

    try {
        if (!continuousFluxFunctionDescriptions.empty()) {
            if (faceInterpolant == nullptr) {
                faceInterpolant = std::make_unique<FaceInterpolant>(subDomain, GetRegion(), faceGeomVec, cellGeomVec);
            }

            faceInterpolant->ComputeRHS(time, locXVec, subDomain->GetAuxVector(), locFVec, GetRegion(), continuousFluxFunctionDescriptions, faceRange, cellGeomVec, faceGeomVec);
        }
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in FaceInterpolant continuousFluxFunctionDescriptions: %s", exception.what());
    }

    RestoreRange(faceRange);
    RestoreRange(cellRange);

    // iterate over any arbitrary RHS functions
    for (const auto& rhsFunction : rhsArbitraryFunctions) {
        PetscCall(rhsFunction.first(*this, subDomain->GetDM(), time, locXVec, locFVec, rhsFunction.second));
    }

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

std::map<std::string, double> ablate::finiteVolume::FiniteVolumeSolver::ComputePhysicsTimeSteps(TS ts) {
    // time steps
    std::map<std::string, double> timeSteps;

    // march over each calculator
    for (const auto& dtFunction : timeStepFunctions) {
        double dt = dtFunction.function(ts, *this, dtFunction.context);
        PetscReal dtMinGlobal;
        MPI_Reduce(&dt, &dtMinGlobal, 1, MPIU_REAL, MPI_MIN, 0, PetscObjectComm((PetscObject)ts)) >> checkMpiError;
        timeSteps[dtFunction.name] = dtMinGlobal;
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
    PetscFunctionBeginUser;
    for (auto& process : processes) {
        if (auto serializablePtr = std::dynamic_pointer_cast<ablate::io::Serializable>(process)) {
            if (serializablePtr->Serialize()) {
                serializablePtr->Save(viewer, sequenceNumber, time);
            }
        }
    }
    PetscFunctionReturnVoid();
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

void ablate::finiteVolume::FiniteVolumeSolver::GetCellRangeWithoutGhost(solver::Range& faceRange) const {
    // Get the point range
    DMLabel solverRegionMinusGhostLabel;
    PetscInt solverRegionMinusGhostValue;
    domain::Region::GetLabel(solverRegionMinusGhost, GetSubDomain().GetDM(), solverRegionMinusGhostLabel, solverRegionMinusGhostValue);

    DMLabelGetStratumIS(solverRegionMinusGhostLabel, solverRegionMinusGhostValue, &faceRange.is) >> checkError;
    if (faceRange.is == nullptr) {
        // There are no points in this region, so skip
        faceRange.start = 0;
        faceRange.end = 0;
        faceRange.points = nullptr;
    } else {
        // Get the range
        ISGetPointRange(faceRange.is, &faceRange.start, &faceRange.end, &faceRange.points) >> checkError;
    }
}
//PetscErrorCode ablate::finiteVolume::FiniteVolumeSolver::ComputeBoundary(PetscReal time, Vec locX, Vec locX_t) {
//    PetscFunctionBeginUser;
//    auto dm = subDomain->GetDM();
//    auto ds = subDomain->GetDiscreteSystem();
//    /* Handle non-essential (e.g. outflow) boundary values.  This should be done before the auxFields are updated so that boundary values can be updated */
//    PetscCall(ablate::solver::Solver::DMPlexInsertBoundaryValues_Plex(dm, ds, PETSC_FALSE, locX, time, faceGeomVec, cellGeomVec, nullptr));
//    PetscFunctionReturn(0);
//
//}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::finiteVolume::FiniteVolumeSolver, "finite volume solver", ARG(std::string, "id", "the name of the flow field"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         ARG(std::vector<ablate::finiteVolume::processes::Process>, "processes", "the processes used to describe the flow"),
         OPT(std::vector<ablate::finiteVolume::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(bool, "computePhysicsTimeStep", "determines if a physics based time step is used to control the FVM time stepping (default is false)"));

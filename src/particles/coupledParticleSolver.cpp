#include "coupledParticleSolver.hpp"

#include <utility>
#include "accessors/eulerianSourceAccessor.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::particles::CoupledParticleSolver::CoupledParticleSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                std::vector<FieldDescription> fields, std::vector<std::shared_ptr<processes::Process>> processesIn,
                                                                std::shared_ptr<initializers::Initializer> initializer, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                                                                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions, const std::vector<std::string>& coupledFields)
    : ParticleSolver(std::move(solverId), std::move(region), std::move(options), std::move(fields), std::move(processesIn), std::move(initializer), std::move(fieldInitialization),
                     std::move(exactSolutions)),
      coupledFieldsNames(coupledFields) {
    // filter through the list of processes for those that are coupled
    coupledProcesses = ablate::utilities::VectorUtilities::Filter<processes::CoupledProcess>(processes);
}

ablate::particles::CoupledParticleSolver::CoupledParticleSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                const std::vector<std::shared_ptr<FieldDescription>>& fields, std::vector<std::shared_ptr<processes::Process>> processes,
                                                                std::shared_ptr<initializers::Initializer> initializer, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                                                                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions, const std::vector<std::string>& coupledFields)
    : CoupledParticleSolver(std::move(solverId), std::move(region), std::move(options), ablate::utilities::VectorUtilities::Copy(fields), std::move(processes), std::move(initializer),
                            std::move(fieldInitialization), std::move(exactSolutions), coupledFields) {}

ablate::particles::CoupledParticleSolver::~CoupledParticleSolver() {
    if (localEulerianSourceVec) {
        VecDestroy(&localEulerianSourceVec) >> utilities::PetscUtilities::checkError;
    }
    if (localEulerianVolumeFactor) {
        VecDestroy(&localEulerianVolumeFactor) >> utilities::PetscUtilities::checkError;
    }
}

/**
 * Map the source terms into the flow field once per time step (They are constant during the time step)
 * @param time
 * @param locX
 * @return
 */
PetscErrorCode ablate::particles::CoupledParticleSolver::PreRHSFunction(TS ts, PetscReal time, bool initialStage, Vec locX) {
    PetscFunctionBeginUser;
    PetscCall(RHSFunction::PreRHSFunction(ts, time, initialStage, locX));

    // march over every coupled field
    for (std::size_t f = 0; f < coupledFields.size(); ++f) {
        const auto& coupledField = coupledFields[f];

        // Create the subDM for only this field
        DM coupledFieldDM;
        IS coupledFieldIS;
        PetscInt fieldId[1] = {coupledField.id};
        PetscCall(DMCreateSubDM(subDomain->GetDM(), 1, fieldId, &coupledFieldIS, &coupledFieldDM));

        // Create a global vector for this subDM to interpolate/push into from the particles
        Vec eulerianFieldSourceVec;
        PetscCall(DMGetGlobalVector(coupledFieldDM, &eulerianFieldSourceVec));

        // project from the particle to the subDM vec
        // project the source terms to the global array
        const char* fieldnames[1] = {coupledParticleFieldsNames[f].c_str()};
        Vec fields[1] = {eulerianFieldSourceVec};
        PetscCall(DMSwarmProjectFields(swarmDm, coupledFieldDM, 1, fieldnames, fields, SCATTER_FORWARD));

        // Bring back to the global source vector
        PetscCall(VecISCopy(localEulerianSourceVec, coupledFieldIS, SCATTER_FORWARD, eulerianFieldSourceVec));

        PetscCall(DMRestoreGlobalVector(coupledFieldDM, &eulerianFieldSourceVec));
        PetscCall(DMDestroy(&coupledFieldDM));
        PetscCall(ISDestroy(&coupledFieldIS));
    }

    // Scale the source vector by the current dt so that when integrated the total is the same
    PetscReal flowTimeStep;
    PetscCall(TSGetTimeStep(ts, &flowTimeStep));
    PetscCall(VecScale(localEulerianSourceVec, 1.0 / flowTimeStep));

    // Scale each source term by the volume of the cell because the project is
    //   M_f u_f = M_p u_p and the M_f includes the volume of the cell
    PetscCall(VecPointwiseMult(localEulerianSourceVec, localEulerianSourceVec, localEulerianVolumeFactor));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * Called to compute the RHS source term for the flow/macro TS
 * @param time
 * @param locX
 * @param locF
 * @return
 */
PetscErrorCode ablate::particles::CoupledParticleSolver::ComputeRHSFunction(PetscReal time, Vec locX, Vec locF) {
    PetscFunctionBeginUser;
    // Add back to the local F vector
    PetscCall(VecAYPX(locF, 1.0, localEulerianSourceVec));
    PetscFunctionReturn(PETSC_SUCCESS);
}

void ablate::particles::CoupledParticleSolver::Setup() {
    // Call the main particle setup
    ParticleSolver::Setup();

    // add a storage location for coupled source terms into the swarm
    if (coupledFieldsNames.empty()) {
        for (const auto& coupledField : subDomain->GetFields()) {
            coupledFields.push_back(coupledField);
        }
    } else {
        for (const auto& coupledFieldName : coupledFieldsNames) {
            const auto& coupledField = subDomain->GetField(coupledFieldName);
            // make sure that it is a solution vector
            if (coupledField.location != domain::FieldLocation::SOL) {
                throw std::invalid_argument("All fields coupled to the flow solver must be domain::FieldLocation::SOL");
            }
            coupledFields.push_back(coupledField);
        }
    }

    // clean/reset the coupled field names
    coupledFieldsNames.clear();

    // for each coupled field create a storage location in the swarm
    for (const auto& field : coupledFields) {
        // Register a new aux field for the source terms to be passed to the main TS
        auto sourceField = FieldDescription{field.name + accessors::EulerianSourceAccessor::CoupledSourceTermPostfix, domain::FieldLocation::AUX, field.components};
        RegisterParticleField(sourceField);

        // store the field names
        coupledParticleFieldsNames.push_back(field.name + accessors::EulerianSourceAccessor::CoupledSourceTermPostfix);
        coupledFieldsNames.push_back(field.name);
    }

    // For coupled simulations we also need to store the previously packed solution, same size as the solution
    const auto& packedSolutionField = GetField(PackedSolution);
    RegisterParticleField(FieldDescription{PreviousPackedSolution, domain::FieldLocation::AUX, packedSolutionField.components});
}

void ablate::particles::CoupledParticleSolver::Initialize() {
    // Call the main particle initialize
    ParticleSolver::Initialize();

    // Get the global vector of the domain we will copy to
    DMCreateLocalVector(subDomain->GetDM(), &localEulerianSourceVec) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(localEulerianSourceVec) >> utilities::PetscUtilities::checkError;

    // duplicate the vec for localEulerianVolumeFactor
    VecDuplicate(localEulerianSourceVec, &localEulerianVolumeFactor) >> utilities::PetscUtilities::checkError;
    VecSet(localEulerianVolumeFactor, 1.0);

    // for FVM fields/meshes march over each cell
    ablate::domain::Range cellRange;
    GetCellRange(cellRange);

    // Get the Array from the localEulerianVolumeFactor
    PetscScalar* localEulerianVolumeFactorArray;
    VecGetArray(localEulerianVolumeFactor, &localEulerianVolumeFactorArray) >> utilities::PetscUtilities::checkError;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        auto cell = cellRange.GetPoint(c);

        // Compute the volume
        PetscReal vol;
        DMPlexComputeCellGeometryFVM(subDomain->GetDM(), c, &vol, nullptr, nullptr) >> utilities::PetscUtilities::checkError;

        // march over each source field
        for (const auto& field : subDomain->GetFields()) {
            if (field.type == domain::FieldType::FVM) {
                // Get each of the stencil pts
                PetscScalar* localEVF;
                DMPlexPointLocalFieldRef(subDomain->GetDM(), cell, field.id, localEulerianVolumeFactorArray, &localEVF) >> utilities::PetscUtilities::checkError;

                if (localEVF) {
                    for (PetscInt n = 0; n < field.numberComponents; ++n) {
                        localEVF[n] = vol;
                    }
                }
            }
        }
    }

    // Cleanup
    VecRestoreArray(localEulerianVolumeFactor, &localEulerianVolumeFactorArray);
    RestoreRange(cellRange);
}

void ablate::particles::CoupledParticleSolver::MacroStepParticles(TS macroTS, bool swarmMigrate) {
    // This function is called after the flow/main TS is advanced so all source terms should have already been added to the flow solver, so reset them to zero here.
    // march over every coupled field
    for (const auto& coupledParticleFieldName : coupledParticleFieldsNames) {
        Vec coupledParticleFieldVec;
        DMSwarmCreateGlobalVectorFromField(GetParticleDM(), coupledParticleFieldName.c_str(), &coupledParticleFieldVec) >> utilities::PetscUtilities::checkError;
        VecZeroEntries(coupledParticleFieldVec) >> utilities::PetscUtilities::checkError;
        DMSwarmDestroyGlobalVectorFromField(GetParticleDM(), coupledParticleFieldName.c_str(), &coupledParticleFieldVec) >> utilities::PetscUtilities::checkError;
    }

    // Before the time step make a copy of the packed solution vector
    Vec packedSolutionVec, previousPackedSolutionVec;
    DMSwarmCreateGlobalVectorFromField(GetParticleDM(), PackedSolution, &packedSolutionVec) >> utilities::PetscUtilities::checkError;
    DMSwarmCreateGlobalVectorFromField(GetParticleDM(), PreviousPackedSolution, &previousPackedSolutionVec) >> utilities::PetscUtilities::checkError;
    VecCopy(packedSolutionVec, previousPackedSolutionVec) >> utilities::PetscUtilities::checkError;
    DMSwarmDestroyGlobalVectorFromField(GetParticleDM(), PackedSolution, &packedSolutionVec) >> utilities::PetscUtilities::checkError;
    DMSwarmDestroyGlobalVectorFromField(GetParticleDM(), PreviousPackedSolution, &previousPackedSolutionVec) >> utilities::PetscUtilities::checkError;

    // Get the start time
    PetscReal startTime;
    TSGetTime(particleTs, &startTime) >> utilities::PetscUtilities::checkError;

    // Call the main time step
    ParticleSolver::MacroStepParticles(macroTS, false);

    // Get the end time
    PetscReal endTime;
    TSGetTime(particleTs, &endTime) >> utilities::PetscUtilities::checkError;

    // Update the source terms
    ComputeEulerianSource(startTime, endTime);

    // Migrate any particles that have moved now that we have done the other calculations
    if (swarmMigrate) {
        SwarmMigrate();
    }
}

void ablate::particles::CoupledParticleSolver::ComputeEulerianSource(PetscReal startTime, PetscReal endTime) {
    Vec packedSolutionVec, previousPackedSolutionVec;

    // extract the vectors again
    DMSwarmCreateGlobalVectorFromField(GetParticleDM(), PackedSolution, &packedSolutionVec) >> utilities::PetscUtilities::checkError;
    DMSwarmCreateGlobalVectorFromField(GetParticleDM(), PreviousPackedSolution, &previousPackedSolutionVec) >> utilities::PetscUtilities::checkError;

    // determine if we should cachePointData
    auto cachePointData = coupledProcesses.size() != 1;

    // compute the source terms for the coupled processes
    // Build the needed data structures
    {  // we need the brackets to force the SwarmAccessor to cleanup before calling the DMSwarmDestroyGlobalVectorFromField cleanup
        accessors::SwarmAccessor swarmAccessorPreStep(cachePointData, swarmDm, fieldsMap, previousPackedSolutionVec);
        accessors::SwarmAccessor swarmAccessorPostStep(cachePointData, swarmDm, fieldsMap, packedSolutionVec);
        accessors::EulerianSourceAccessor eulerianSourceAccessor(cachePointData, swarmDm, fieldsMap);

        for (auto& coupledProcess : coupledProcesses) {
            coupledProcess->ComputeEulerianSource(startTime, endTime, swarmAccessorPreStep, swarmAccessorPostStep, eulerianSourceAccessor);
        }
    }

    DMSwarmDestroyGlobalVectorFromField(GetParticleDM(), PackedSolution, &packedSolutionVec) >> utilities::PetscUtilities::checkError;
    DMSwarmDestroyGlobalVectorFromField(GetParticleDM(), PreviousPackedSolution, &previousPackedSolutionVec) >> utilities::PetscUtilities::checkError;
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::particles::CoupledParticleSolver, "Coupled Lagrangian particle solver", ARG(std::string, "id", "the name of the particle solver"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         OPT(std::vector<ablate::particles::FieldDescription>, "fields", "any additional fields beside coordinates"),
         ARG(std::vector<ablate::particles::processes::Process>, "processes", "the processes used to describe the particle source terms"),
         ARG(ablate::particles::initializers::Initializer, "initializer", "the initial particle setup methods"),
         OPT(std::vector<ablate::mathFunctions::FieldFunction>, "fieldInitialization", "the initial particle fields values"),
         OPT(std::vector<ablate::mathFunctions::FieldFunction>, "exactSolutions", "particle fields (SOL) exact solutions"),
         OPT(std::vector<std::string>, "coupledFields", "list of fields to couple with Eulerian TS.  If empty or not specified all fields are coupled."));
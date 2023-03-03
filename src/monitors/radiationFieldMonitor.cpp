#include "radiationFieldMonitor.hpp"

ablate::monitors::RadiationFieldMonitor::RadiationFieldMonitor(const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                                                               std::shared_ptr<io::interval::Interval> intervalIn)
    : eos(eosIn), radiationModel(radiationModelIn), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {}

void ablate::monitors::RadiationFieldMonitor::Register(std::shared_ptr<ablate::solver::Solver> solverIn) {
    FieldMonitor::Register(solverIn);

    // Create the monitor name
    std::string dmID = "radiationFieldMonitor";

    std::vector<std::shared_ptr<domain::FieldDescriptor>> fields(fieldNames.size(), nullptr);

    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        fields[f] = std::make_shared<domain::FieldDescription>(fieldNames[f], fieldNames[f], domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM);
    }

    // Register all fields with the monitorDomain
    ablate::monitors::FieldMonitor::Register(dmID, solverIn, fields);

    // Get the density thermodynamic function
    absorptivityFunction = radiationModel->GetAbsorptionPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, solverIn->GetSubDomain().GetFields());
}

PetscErrorCode ablate::monitors::RadiationFieldMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    // Perform the principal save
    PetscCall(ablate::monitors::FieldMonitor::Save(viewer, sequenceNumber, time));

    // Save the step number
    PetscCall(ablate::io::Serializable::SaveKeyValue(viewer, "step", step));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::monitors::RadiationFieldMonitor::MonitorRadiation(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;

    // Loads in context
    auto monitor = (ablate::monitors::RadiationFieldMonitor*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
        // Increment the number of steps taken so far
        monitor->step += 1;

        // Extract the main solution vector for the absorptivity calculation
        DM solDM;
        Vec solVec;
        solVec = monitor->GetSolver()->GetSubDomain().GetSolutionVector();
        solDM = monitor->GetSolver()->GetSubDomain().GetDM();

        /**
         * We require the aux DM to extract temperature information.
         */
        DM auxDm;
        Vec auxVec;
        auxVec = monitor->GetSolver()->GetSubDomain().GetAuxVector();
        auxDm = monitor->GetSolver()->GetSubDomain().GetAuxDM();

        // Store the monitorDM, monitorVec, and the monitorFields
        DM monitorDM = monitor->monitorSubDomain->GetDM();
        Vec monitorVec = monitor->monitorSubDomain->GetSolutionVector();
        auto& monitorFields = monitor->monitorSubDomain->GetFields();

        // Get the local cell range
        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(monitor->monitorSubDomain->GetDM(), 0, &cStart, &cEnd);

        //! Get the local to global cell mapping. This ensures that the cell index mapping between the monitor DM and the global solution is correct
        IS subpointIS;
        const PetscInt* subpointIndices;
        DMPlexGetSubpointIS(monitorDM, &subpointIS);
        ISGetIndices(subpointIS, &subpointIndices);

        // Extract the solution global array
        const PetscScalar* solArray;
        PetscCall(VecGetArrayRead(solVec, &solArray));

        // Extract the aux global array
        const PetscScalar* auxArray;
        PetscCall(VecGetArrayRead(auxVec, &auxArray));

        // Extract the monitor array
        PetscScalar* monitorArray;
        PetscCall(VecGetArray(monitorVec, &monitorArray));

        const auto& temperatureFieldInfo = monitor->GetSolver()->GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD);

        // Get the timestep from the TS
        PetscReal dt;
        PetscCall(TSGetTimeStep(ts, &dt));

        // Create pointers to the field and monitor data exterior to loop
        double kappa = 1;                  //!< Absorptivity coefficient, property of each cell
        PetscReal* temperature = nullptr;  //!< The temperature at any given location
        const PetscScalar* solPt;
        PetscScalar* monitorPt;

        //! Compute measures
        for (PetscInt c = cStart; c < cEnd; c++) {
            PetscInt monitorCell = c;
            PetscInt masterCell = subpointIndices[monitorCell];  //! Gets the cell index associated with this position in the monitor DM cell range?

            // Get solution point data
            PetscCall(DMPlexPointLocalRead(solDM, masterCell, solArray, &solPt));

            // Get read/write access to point in monitor array
            PetscCall(DMPlexPointLocalRef(monitorDM, monitorCell, monitorArray, &monitorPt));

            if (monitorPt && solPt) {
                // compute absorptivity
                PetscCall(DMPlexPointLocalFieldRead(auxDm, masterCell, temperatureFieldInfo.id, auxArray, &temperature));

                // Get the absorptivity data from solution point data
                PetscCall(monitor->absorptivityFunction.function(solPt, *temperature, &kappa, monitor->absorptivityFunction.context.get()));

                // Perform actual calculations now
                monitorPt[monitorFields[FieldPlacements::intensity].offset] = kappa * ablate::utilities::Constants::sbc * *temperature * *temperature * *temperature * *temperature;
                monitorPt[monitorFields[FieldPlacements::absorption].offset] = kappa;
            }
        }
        PetscCall(VecRestoreArray(monitorVec, &monitorArray));

        // Cleanup
        // Restore arrays
        PetscCall(ISRestoreIndices(subpointIS, &subpointIndices));
        PetscCall(VecRestoreArrayRead(solVec, &solArray));
        PetscCall(VecRestoreArray(monitorVec, &monitorArray));
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::RadiationFieldMonitor, "A solver for radiative heat transfer in participating media", ARG(ablate::eos::EOS, "eos", "The equation of state"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "properties model for the output of radiation properties within the field"),
         OPT(ablate::io::interval::Interval, "interval", "The monitor output interval"));
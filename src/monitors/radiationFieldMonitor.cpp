#include "radiationFieldMonitor.hpp"

ablate::monitors::RadiationFieldMonitor::RadiationFieldMonitor(const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                                                               std::shared_ptr<io::interval::Interval> intervalIn)
    : eos(eosIn), radiationModel(radiationModelIn), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {}

void ablate::monitors::RadiationFieldMonitor::Register(std::shared_ptr<ablate::solver::Solver> solverIn) {
    Monitor::Register(solverIn);

    // Create the monitor name
    std::string dmID = "radiationFieldMonitor";

    std::vector<std::shared_ptr<domain::FieldDescriptor>> fields(fieldNames.size(), nullptr);

    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        fields[f] = std::make_shared<domain::FieldDescription>(fieldNames[f], fieldNames[f], domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM);
    }

    // Register all fields with the monitorDomain
    ablate::monitors::FieldMonitor::Register(dmID, solverIn, fields);

    // Get the density thermodynamic function
    absorptivityFunction = radiationModel->GetRadiationPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, solverIn->GetSubDomain().GetFields());
}

void ablate::monitors::RadiationFieldMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // Perform the principal save
    ablate::monitors::FieldMonitor::Save(viewer, sequenceNumber, time);

    // Save the step number
    ablate::io::Serializable::SaveKeyValue(viewer, "step", step);
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
        solVec = monitor->GetSolver()->GetSubDomain().GetSubSolutionVector();
        solDM = monitor->GetSolver()->GetSubDomain().GetDM();

        /**
         * We require the aux DM to extract temperature information.
         */
        DM auxDm;
        Vec auxVec;
        auxVec = monitor->GetSolver()->GetSubDomain().GetAuxVector();
        auxDm = monitor->GetSolver()->GetSubDomain().GetAuxDM();

        // Store the monitorDM, monitorVec, and the monitorFields
        DM monitorDM = monitor->monitorSubDomain->GetSubDM();
        Vec monitorVec = monitor->monitorSubDomain->GetSolutionVector();
        auto& monitorFields = monitor->monitorSubDomain->GetFields();

        /**
         * The monitor needs to read the information from the monitor DM to know which points it needs to write to
         */
        //        for (std::size_t f = 0; f < monitor->fieldNames.size(); f++) {
        //            const auto& field = monitor->monitorSubDomain->GetField(monitor->fieldNames[f]);
        //            monitor->GetSolver()->GetSubDomain().GetFieldGlobalVector(field, &vecIS[f], &vec[f], &fieldDM[f]) >> utilities::PetscUtilities::checkError;
        //        }

        // Get the local cell range
        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(monitor->monitorSubDomain->GetSubDM(), 0, &cStart, &cEnd);

        //! Get the local to global cell mapping. This ensures that the cell index mapping between the monitor DM and the global solution is correct
        IS subpointIS;
        const PetscInt* subpointIndices;
        DMPlexGetSubpointIS(monitorDM, &subpointIS);
        ISGetIndices(subpointIS, &subpointIndices);

        // Extract the solution global array
        const PetscScalar* solDat;
        VecGetArrayRead(solVec, &solDat) >> utilities::PetscUtilities::checkError;

        // Extract the aux global array
        const PetscScalar* auxDat;
        VecGetArrayRead(auxVec, &auxDat) >> utilities::PetscUtilities::checkError;

        // Extract the monitor array
        PetscScalar* monitorDat;
        VecGetArray(monitorVec, &monitorDat) >> utilities::PetscUtilities::checkError;

        const auto& temperatureFieldInfo = monitor->GetSolver()->GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::TEMPERATURE_FIELD);

        // Get the timestep from the TS
        PetscReal dt;
        TSGetTimeStep(ts, &dt) >> utilities::PetscUtilities::checkError;

        // Create pointers to the field and monitor data exterior to loop
        const PetscScalar* fieldDat;
        double kappa = 1;                  //!< Absorptivity coefficient, property of each cell
        PetscReal* temperature = nullptr;  //!< The temperature at any given location
//        const PetscScalar* fieldPt;
        const PetscScalar* solPt;

        PetscScalar* monitorPt;

        for (std::size_t f = 0; f < monitor->fieldNames.size(); f++) {
            // Extract the field vector global array
            VecGetArrayRead(monitorVec, &fieldDat) >> utilities::PetscUtilities::checkError;  // TODO: Do we need an offset here?

            //! Compute measures
            for (PetscInt c = cStart; c < cEnd; c++) {
                PetscInt monitorCell = c;
                PetscInt masterCell = subpointIndices[monitorCell];  //! Gets the cell index associated with this position in the monitor DM cell range?

                // TODO: We don't need to extract any information from the monitor field to get the new values.

                //                // Get field point data
                //                DMPlexPointLocalFieldRead(monitorDM, cellSegment.cell, temperatureField.id, auxArray, &temperature);
                //                DMPlexPointLocalRead(fieldDM[f], masterCell, fieldDat, &fieldPt) >> utilities::PetscUtilities::checkError;

                // Get solution point data
                DMPlexPointLocalRead(solDM, masterCell, solDat, &solPt) >> utilities::PetscUtilities::checkError;

                // Get read/write access to point in monitor array
                DMPlexPointGlobalRef(monitorDM, monitorCell, monitorDat, &monitorPt) >> utilities::PetscUtilities::checkError;

                if (monitorPt && solPt) {
                    // compute absorptivity
                    DMPlexPointLocalFieldRead(auxDm, masterCell, temperatureFieldInfo.id, auxDat, &temperature) >> utilities::PetscUtilities::checkError;

                    // Get the absorptivity data from solution point data
                    monitor->absorptivityFunction.function(solPt, *temperature, &kappa, monitor->absorptivityFunction.context.get());

                    // Perform actual calculations now
                    monitorPt[monitorFields[FieldPlacements::intensity].offset] += kappa * ablate::utilities::Constants::sbc * *temperature * *temperature * *temperature * *temperature;
                    monitorPt[monitorFields[FieldPlacements::absorption].offset] += kappa;
                }
            }
            VecRestoreArrayRead(monitorVec, &fieldDat) >> utilities::PetscUtilities::checkError;  // TODO: Do we need an offset here?
        }
        // Cleanup
        // Restore arrays
        ISRestoreIndices(subpointIS, &subpointIndices) >> utilities::PetscUtilities::checkError;
        VecRestoreArrayRead(solVec, &solDat) >> utilities::PetscUtilities::checkError;
        VecRestoreArray(monitorVec, &monitorDat) >> utilities::PetscUtilities::checkError;
        //        // Restore field vectors
        //        for (std::size_t f = 0; f < monitor->fieldNames.size(); f++) {
        //            const auto& field = monitor->GetSolver()->GetSubDomain().GetField(monitor->fieldNames[f]);
        //            monitor->GetSolver()->GetSubDomain().RestoreFieldGlobalVector(field, &vecIS[f], &vec[f], &fieldDM[f]) >> utilities::PetscUtilities::checkError;
        //        }
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::RadiationFieldMonitor, "A solver for radiative heat transfer in participating media", ARG(ablate::eos::EOS, "eos", "The equation of state"),
         ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "properties model for the output of radiation properties within the field"),
         OPT(ablate::io::interval::Interval, "interval", "The monitor output interval"));
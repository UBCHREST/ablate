#include "monitors/turbFlowStats.hpp"
#include <iostream>
#include "io/interval/fixedInterval.hpp"
#include "petscmath.h"
#include "solver/range.hpp"
#include "utilities/constants.hpp"

using tp = ablate::eos::ThermodynamicProperty;
using fLoc = ablate::domain::FieldLocation;
using Constant = ablate::utilities::Constants;
typedef ablate::solver::Range Range;

ablate::monitors::TurbFlowStats::TurbFlowStats(const std::vector<std::string> nameIn, const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<io::interval::Interval> intervalIn)
    : fieldNames(nameIn), eos(eosIn), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {
    step = 0;
}

PetscErrorCode ablate::monitors::TurbFlowStats::MonitorTurbFlowStats(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Loads in context
    auto monitor = (ablate::monitors::TurbFlowStats*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
        // Increment the number of steps taken so far
        monitor->step += 1;

        // Extract all fields to be monitored
        std::vector<Vec> vec(monitor->fieldNames.size(), nullptr);
        std::vector<IS> vecIS(monitor->fieldNames.size(), nullptr);
        std::vector<DM> fieldDM(monitor->fieldNames.size(), nullptr);
        for (std::size_t f = 0; f < monitor->fieldNames.size(); f++) {
            const auto& field = monitor->GetSolver()->GetSubDomain().GetField(monitor->fieldNames[f]);
            ierr = monitor->GetSolver()->GetSubDomain().GetFieldGlobalVector(field, &vecIS[f], &vec[f], &fieldDM[f]);
            CHKERRQ(ierr);
        }

        // Extract the main solution vector for the density calculation
        DM solDM;
        Vec solVec;
        solVec = monitor->GetSolver()->GetSubDomain().GetSubSolutionVector();
        solDM = monitor->GetSolver()->GetSubDomain().GetDM();

        // Store the monitorDM, monitorVec, and the monitorFields
        DM monitorDM = monitor->monitorSubDomain->GetSubDM();
        Vec monitorVec = monitor->monitorSubDomain->GetSolutionVector();
        auto& monitorFields = monitor->monitorSubDomain->GetFields();

        // Get the local cell range
        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(monitor->monitorSubDomain->GetSubDM(), 0, &cStart, &cEnd);

        // Get the local to global cell mapping
        IS subpointIS;
        const PetscInt* subpointIndices;
        DMPlexGetSubpointIS(monitorDM, &subpointIS);
        ISGetIndices(subpointIS, &subpointIndices);

        // Extract the solution global array
        const PetscScalar* solDat;
        ierr = VecGetArrayRead(solVec, &solDat);
        CHKERRQ(ierr);

        // Extract the monitor array
        PetscScalar* monitorDat;
        ierr = VecGetArray(monitorVec, &monitorDat);
        CHKERRQ(ierr);

        // Get the timestep from the TS
        PetscReal dt;
        ierr = TSGetTimeStep(ts, &dt);
        CHKERRQ(ierr);

        // Create pointers to the field and monitor data exterior to loop
        const PetscScalar* fieldDat;
        //! Iterator guide
        // f - field iterator
        // c - cell iterator
        // p - field component iterator
        for (std::size_t f = 0; f < monitor->fieldNames.size(); f++) {
            // Extract the field vector global array
            ierr = VecGetArrayRead(vec[f], &fieldDat);
            CHKERRQ(ierr);

            //! Compute measures
            for (PetscInt c = cStart; c < cEnd; c++) {
                PetscInt monitorCell = c;
                PetscInt masterCell = subpointIndices[monitorCell];
                const PetscScalar* fieldPt;
                const PetscScalar* solPt;
                PetscScalar* monitorPt;

                // Get field point data
                ierr = DMPlexPointLocalRead(fieldDM[f], masterCell, fieldDat, &fieldPt);
                CHKERRQ(ierr);

                // Get solution point data
                ierr = DMPlexPointLocalRead(solDM, masterCell, solDat, &solPt);
                CHKERRQ(ierr);

                // Get read/write access to point in monitor array
                ierr = DMPlexPointGlobalRef(monitorDM, monitorCell, monitorDat, &monitorPt);
                CHKERRQ(ierr);

                if (monitorPt && solPt) {
                    // Get the density data from solution point data
                    PetscReal densLoc;
                    monitor->densityFunc.function(solPt, &densLoc, monitor->densityFunc.context.get());

                    // Perform actual calculations now
                    monitorPt[monitorFields[FieldPlacements::densitySum].offset] += densLoc;
                    monitorPt[monitorFields[FieldPlacements::densityDtSum].offset] += densLoc * dt;

                    // Get fields to be monitored here
                    const auto& field = monitor->GetSolver()->GetSubDomain().GetField(monitor->fieldNames[f]);

                    // March over each field component
                    for (int p = 0; p < field.numberComponents; p++) {
                        // Set the offset. The first two offset places are reserved for "field placements". Each field component takes " SectionLabels::END" number of offset placements.
                        // the next field starts the offset counter from where the last field stops.
                        PetscInt offset = SectionLabels::END * p + monitorFields[FieldPlacements::fieldsStart + f].offset;

                        monitorPt[offset + SectionLabels::densityMult] += fieldPt[p] * densLoc;
                        monitorPt[offset + SectionLabels::densityDtMult] += fieldPt[p] * densLoc * dt;
                        monitorPt[offset + SectionLabels::densitySqr] += fieldPt[p] * fieldPt[p] * densLoc;
                        monitorPt[offset + SectionLabels::sum] += fieldPt[p];
                        monitorPt[offset + SectionLabels::sumSqr] += fieldPt[p] * fieldPt[p];
                        monitorPt[offset + SectionLabels::favreAvg] =
                            monitorPt[offset + SectionLabels::densityDtMult] / (monitorPt[monitorFields[FieldPlacements::densityDtSum].offset] + Constant::tiny);
                        monitorPt[offset + SectionLabels::rms] = PetscSqrtReal(monitorPt[offset + SectionLabels::sumSqr] / (monitor->step + Constant::tiny) -
                                                                               PetscPowReal(monitorPt[offset + SectionLabels::sum] / (monitor->step + Constant::tiny), 2));
                        monitorPt[offset + SectionLabels::mRms] =
                            PetscSqrtReal(monitorPt[offset + SectionLabels::densitySqr] / (monitorPt[monitorFields[FieldPlacements::densitySum].offset] + Constant::tiny) -
                                          PetscPowReal(monitorPt[offset + SectionLabels::densityMult] / (monitorPt[monitorFields[FieldPlacements::densitySum].offset] + Constant::tiny), 2));
                    }
                }
            }
            ierr = VecRestoreArrayRead(vec[f], &fieldDat);
            CHKERRQ(ierr);
        }
        // Cleanup
        // Restore arrays
        ierr = ISRestoreIndices(subpointIS, &subpointIndices);
        CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(solVec, &solDat);
        CHKERRQ(ierr);
        ierr = VecRestoreArray(monitorVec, &monitorDat);
        CHKERRQ(ierr);
        // Restore field vectors
        for (std::size_t f = 0; f < monitor->fieldNames.size(); f++) {
            const auto& field = monitor->GetSolver()->GetSubDomain().GetField(monitor->fieldNames[f]);
            ierr = monitor->GetSolver()->GetSubDomain().RestoreFieldGlobalVector(field, &vecIS[f], &vec[f], &fieldDM[f]);
            CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}

void ablate::monitors::TurbFlowStats::Register(std::shared_ptr<ablate::solver::Solver> solverIn) {
    // Create the monitor name
    std::string dmID = solverIn->GetSolverId() + "_turbulenceFlowStats";

    // Create suffix vector
    std::vector<std::string> suffix{"rhoMult", "rhoDtMult", "rhoSqr", "sum", "sumSqr", "favreAvg", "rms", "mRms"};

    // Create vectors of names for all components of all fields
    std::vector<std::vector<std::string>> processedCompNames;
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        const auto& field = solverIn->GetSubDomain().GetField(fieldNames[f]);
        std::vector<std::string> innerCompNames(SectionLabels::END * field.numberComponents);
        for (PetscInt c = 0; c < field.numberComponents; c++) {
            for (std::size_t p = 0; p < suffix.size(); p++) {
                innerCompNames[SectionLabels::END * c + p] = (field.numberComponents > 1 ? field.components[c] + "_" : "") + suffix[p];
            }
        }
        processedCompNames.push_back(innerCompNames);
    }

    // Create all FieldDescription objects
    std::vector<std::shared_ptr<domain::FieldDescriptor>> fields(fieldNames.size() + FieldPlacements::fieldsStart, nullptr);
    fields[FieldPlacements::densitySum] =
        std::make_shared<domain::FieldDescription>("densitySum", "densitySum", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM);
    fields[FieldPlacements::densityDtSum] =
        std::make_shared<domain::FieldDescription>("densityDtSum", "densityDtSum", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM);

    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        fields[FieldPlacements::fieldsStart + f] = std::make_shared<domain::FieldDescription>(fieldNames[f], fieldNames[f], processedCompNames[f], domain::FieldLocation::SOL, domain::FieldType::FVM);
    }

    // Register all fields with the monitorDomain
    ablate::monitors::FieldMonitor::Register(dmID, solverIn, fields);

    // Get the density thermodynamic function
    densityFunc = eos->GetThermodynamicFunction(tp::Density, this->GetSolver()->GetSubDomain().GetFields(fLoc::SOL));
}

void ablate::monitors::TurbFlowStats::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // Perform the principal save
    ablate::monitors::FieldMonitor::Save(viewer, sequenceNumber, time);

    // Save the step number
    ablate::io::Serializable::SaveKeyValue(viewer, "step", step);
}

void ablate::monitors::TurbFlowStats::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // Perform the principal restore
    ablate::monitors::FieldMonitor::Restore(viewer, sequenceNumber, time);

    // Restore the step number
    ablate::io::Serializable::RestoreKeyValue(viewer, "step", step);
}

#include <registrar.hpp>
REGISTER(ablate::monitors::Monitor, ablate::monitors::TurbFlowStats, "Computes turbulent flow statistics", ARG(std::vector<std::string>, "fields", "The name of the field"),
         ARG(ablate::eos::EOS, "eos", "The equation of state"), OPT(ablate::io::interval::Interval, "interval", "The monitor output interval"));
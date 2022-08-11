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
    : fieldNames(nameIn), eos(eosIn), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {}

ablate::monitors::TurbFlowStats::~TurbFlowStats() {
    VecDestroy(&turbVec);
    DMDestroy(&turbDM);
}

PetscErrorCode ablate::monitors::TurbFlowStats::MonitorTurbFlowStats(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Loads in context
    auto monitor = (ablate::monitors::TurbFlowStats*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
        std::vector<Vec> vec(monitor->fieldNames.size(), nullptr);
        std::vector<IS> vecIS(monitor->fieldNames.size(), nullptr);
        std::vector<DM> subDM(monitor->fieldNames.size(), nullptr);
        for (std::size_t f = 0; f < monitor->fieldNames.size(); f++) {
            const auto& field = monitor->GetSolver()->GetSubDomain().GetField(monitor->fieldNames[f]);

            ierr = monitor->GetSolver()->GetSubDomain().GetFieldGlobalVector(field, &vecIS[f], &vec[f], &subDM[f]);
            CHKERRQ(ierr);
        }

        DM solDM;
        Vec solVec;
        solVec = monitor->GetSolver()->GetSubDomain().GetSubSolutionVector();
        solDM = monitor->GetSolver()->GetSubDomain().GetDM();

        //! Get relevant data
        // Get cell range (just keep data in this struct and read out as necessary).
        Range cellRange;
        monitor->GetSolver()->GetCellRange(cellRange);

        // Extract the solution global array
        const PetscScalar* solDat;
        ierr = VecGetArrayRead(solVec, &solDat);
        CHKERRQ(ierr);

        // Get the array of turbulent flow data
        PetscScalar* turbDat;
        ierr = VecGetArray(monitor->turbVec, &turbDat);
        CHKERRQ(ierr);

        // Get the timestep from the TS
        PetscReal dt;
        ierr = TSGetTimeStep(ts, &dt);
        CHKERRQ(ierr);

        //! Iterator guide
        // f - field iterator
        // c - cell iterator
        // p - field component iterator
        const PetscScalar* fieldDat;
        for (std::size_t f = 0; f < monitor->fieldNames.size(); f++) {
            // Extract the field vector global array
            ierr = VecGetArrayRead(vec[f], &fieldDat);
            CHKERRQ(ierr);

            //! Compute measures
            for (PetscInt c = cellRange.start; c < cellRange.end; c++) {
                const PetscScalar* fieldPt;
                const PetscScalar* solPt;
                PetscScalar* turbPt;

                // Get field point data
                ierr = DMPlexPointLocalRead(subDM[f], cellRange.points[c], fieldDat, &fieldPt);
                CHKERRQ(ierr);

                // Get solution point data
                ierr = DMPlexPointLocalRead(solDM, cellRange.points[c], solDat, &solPt);
                CHKERRQ(ierr);

                // Get read/write access to point in turbulent flow array
                ierr = DMPlexPointGlobalRef(monitor->turbDM, cellRange.points[c], turbDat, &turbPt);
                CHKERRQ(ierr);

                if (turbPt) {
                    // Get the density data from solution point data
                    PetscReal densLoc;
                    monitor->densityFunc.function(solPt, &densLoc, monitor->densityFunc.context.get());

                    // Perform actual calculations now
                    turbPt[monitor->CatOffset.densitySum] += densLoc;
                    turbPt[monitor->CatOffset.densityDtSum] += densLoc * dt;

                    // March over each field component
                    for (int p = monitor->fieldTrack[f]; p < monitor->fieldTrack[f + 1]; p++) {
                        turbPt[monitor->CatOffset.densityMult + p] += fieldPt[p] * densLoc;
                        turbPt[monitor->CatOffset.densityDtMult + p] += fieldPt[p] * densLoc * dt;
                        turbPt[monitor->CatOffset.densitySqr + p] += fieldPt[p] * fieldPt[p] * densLoc;
                        turbPt[monitor->CatOffset.sum + p] += fieldPt[p];
                        turbPt[monitor->CatOffset.sumSqr + p] += fieldPt[p] * fieldPt[p];
                        turbPt[monitor->CatOffset.favreAvg + p] += turbPt[monitor->CatOffset.densityDtMult + p] / (turbPt[monitor->CatOffset.densityDtSum] + Constant::tiny);
                        turbPt[monitor->CatOffset.rms + p] =
                            PetscSqrtReal(turbPt[monitor->CatOffset.sumSqr + p] / (step + Constant::tiny) - PetscPowReal(turbPt[monitor->CatOffset.sum + p] / (step + Constant::tiny), 2));
                        turbPt[monitor->CatOffset.mRms + p] = PetscSqrtReal(turbPt[monitor->CatOffset.densitySqr + p] / (turbPt[monitor->CatOffset.densitySum] + Constant::tiny) -
                                                                            PetscPowReal(turbPt[monitor->CatOffset.densityMult + p] / (turbPt[monitor->CatOffset.densitySum] + Constant::tiny), 2));
                    }
                }
            }
        }
        // Free the turbulent flow array
        ierr = VecRestoreArray(monitor->turbVec, &turbDat);
        CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

void ablate::monitors::TurbFlowStats::AddField(DM& dm, const char* nameField, PetscInt numComp) {
    PetscFV fvm;
    PetscFVCreate(PetscObjectComm(PetscObject(dm)), &fvm) >> checkError;
    PetscObjectSetName((PetscObject)fvm, nameField) >> checkError;
    PetscFVSetFromOptions(fvm) >> checkError;
    PetscFVSetNumComponents(fvm, numComp) >> checkError;

    DMAddField(dm, nullptr, (PetscObject)fvm) >> checkError;
    PetscFVDestroy(&fvm);
}

void ablate::monitors::TurbFlowStats::Register(std::shared_ptr<ablate::solver::Solver> solverIn) {
    // Copy over the solver
    ablate::monitors::Monitor::Register(solverIn);

    // Copy the master DM over, set fields, and initialize vector values
    this->GetSolver()->GetSubDomain().CheckSubDM(&turbDM);
    std::string DmName = "TurbDM";
    PetscObjectSetName((PetscObject)turbDM, DmName.c_str());

    // Add the densitySum and densityDtSum fields
    std::string densitySum = "rhoSum";
    AddField(turbDM, densitySum.c_str(), 1);

    std::string densityDtSum = "rhoDtSum";
    AddField(turbDM, densityDtSum.c_str(), 1);

    // Initialize the numComps variable, which will store the sum of all the fields
    PetscInt numComp = 0;

    // Resize the fieldTrack and fieldComps vectors to be the proper size (fieldTrack one greater than num fields, fieldComps equal to num fields)
    fieldComps.resize(fieldNames.size(), 0);
    fieldTrack.resize(fieldNames.size() + 1, 0);
    fieldTrack[0] = 0;

    // Need to find the total number of field components and the indices marking the beginning of each field
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        const auto& field = this->GetSolver()->GetSubDomain().GetField(fieldNames[f]);

        fieldComps[f] = field.numberComponents;
        numComp += field.numberComponents;
        if ((f + 1) < fieldTrack.size()) {
            fieldTrack[f + 1] = fieldTrack[f] + field.numberComponents;
        }
    }

    // Set the mode of the category offset and the category order
    CatOffset.SetMode(numComp);
    CatOrder.SetMode(fieldNames.size());

    // Add all fields to the rhoMult category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        std::string densityMult = "rhoMult_" + fieldNames[f];
        AddField(turbDM, densityMult.c_str(), fieldComps[f]);
    }

    // Add all fields to the rhoDtMult category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        std::string densityDtMult = "rhoDtMult_" + fieldNames[f];
        AddField(turbDM, densityDtMult.c_str(), fieldComps[f]);
    }

    // Add all fields to the rhoSqr category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        std::string densitySqr = "rhoSqr_" + fieldNames[f];
        AddField(turbDM, densitySqr.c_str(), fieldComps[f]);
    }

    // Add all fields to the sum category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        std::string sum = "sum_" + fieldNames[f];
        AddField(turbDM, sum.c_str(), fieldComps[f]);
    }

    // Add all fields to the sumSqr category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        std::string sumSqr = "sumSqr_" + fieldNames[f];
        AddField(turbDM, sumSqr.c_str(), fieldComps[f]);
    }

    // Add all fields to the favreAvg category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        std::string favreAvg = "favreAvg_" + fieldNames[f];
        AddField(turbDM, favreAvg.c_str(), fieldComps[f]);
    }

    // Add all fields to the rms category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        std::string rms = "rms_" + fieldNames[f];
        AddField(turbDM, rms.c_str(), fieldComps[f]);
    }

    // Add all fields to the mRms category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        std::string mRms = "mRms_" + fieldNames[f];
        AddField(turbDM, mRms.c_str(), fieldComps[f]);
    }

    // Create the PetscSection
    PetscSection turbSection;
    DMGetLocalSection(turbDM, &turbSection);

    // Register densitySum and densityDtSum with the PetscSection
    PetscSectionSetComponentName(turbSection, CatOrder.densitySum, 0, densitySum.c_str());
    PetscSectionSetComponentName(turbSection, CatOrder.densityDtSum, 0, densityDtSum.c_str());

    // Register all component-number-dependent fields with their proper category
    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        const auto& field = this->GetSolver()->GetSubDomain().GetField(fieldNames[f]);
        for (int p = 0; p < field.numberComponents; p++) {
            PetscSectionSetComponentName(turbSection, CatOrder.densityMult + f, p, field.components[p].c_str());
            PetscSectionSetComponentName(turbSection, CatOrder.densityDtMult + f, p, field.components[p].c_str());
            PetscSectionSetComponentName(turbSection, CatOrder.densitySqr + f, p, field.components[p].c_str());
            PetscSectionSetComponentName(turbSection, CatOrder.sum + f, p, field.components[p].c_str());
            PetscSectionSetComponentName(turbSection, CatOrder.sumSqr + f, p, field.components[p].c_str());
            PetscSectionSetComponentName(turbSection, CatOrder.favreAvg + f, p, field.components[p].c_str());
            PetscSectionSetComponentName(turbSection, CatOrder.rms + f, p, field.components[p].c_str());
            PetscSectionSetComponentName(turbSection, CatOrder.mRms + f, p, field.components[p].c_str());
        }
    }

    //! Replace name with id here
    // Name the PetscSection
    PetscObjectSetName((PetscObject)turbSection, GetId().c_str());

    DMCreateGlobalVector(turbDM, &turbVec);

    std::string turbName = "Turb";
    PetscObjectSetName((PetscObject)turbVec, turbName.c_str());
    VecSet(turbVec, 0.0);

    // Get the density thermodynamic function
    densityFunc = eos->GetThermodynamicFunction(tp::Density, this->GetSolver()->GetSubDomain().GetFields(fLoc::SOL));
}

void ablate::monitors::TurbFlowStats::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    if (sequenceNumber == 0) {
        DMView(turbDM, viewer) >> checkError;
    }

    DMSetOutputSequenceNumber(turbDM, sequenceNumber, time);

    VecView(turbVec, viewer) >> checkError;
}

void ablate::monitors::TurbFlowStats::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    DMSetOutputSequenceNumber(turbDM, sequenceNumber, time);
    VecLoad(turbVec, viewer) >> checkError;
}

#include <registrar.hpp>
REGISTER(ablate::monitors::Monitor, ablate::monitors::TurbFlowStats, "Computes turbulent flow statistics", ARG(std::vector<std::string>, "fields", "The name of the field"),
         ARG(ablate::eos::EOS, "eos", "The equation of state"), OPT(ablate::io::interval::Interval, "interval", "The monitor output interval"));
#include "monitors/turbFlowStats.hpp"
#include "monitors/logs/stdOut.hpp"
#include "io/interval/fixedInterval.hpp"
#include "petscmath.h"
#include "solver/range.hpp"

using ttf = ablate::eos::ThermodynamicFunction;
using tp = ablate::eos::ThermodynamicProperty;
using fLoc =  ablate::domain::FieldLocation;
typedef ablate::solver::Range Range;

ablate::monitors::TurbFlowStats::TurbFlowStats(const std::string& nameIn, const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<logs::Log> logIn, std::shared_ptr<io::interval::Interval> intervalIn)
    : fieldName(nameIn), eos(eosIn), log(logIn ? logIn : std::make_shared<logs::StdOut>()), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()){}

PetscErrorCode ablate::monitors::TurbFlowStats::MonitorTurbFlowStats(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Loads in context
    auto monitor = (ablate::monitors::TurbFlowStats*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
        const auto& field = monitor->GetSolver()->GetSubDomain().GetField(monitor->fieldName);
        IS vecIS;
        DM subDM, solDM;
        Vec vec, solVec;

        ierr = monitor->GetSolver()->GetSubDomain().GetFieldGlobalVector(field, &vecIS, &vec, &subDM);
        CHKERRQ(ierr);

        solVec = monitor->GetSolver()->GetSubDomain().GetSubSolutionVector();
        solDM = monitor->GetSolver()->GetSubDomain().GetDM();

        MPI_Comm comm = PetscObjectComm((PetscObject)vec);
        if (monitor->log->Initialized()) {
            monitor->log->Initialize(comm);
        }

        //! Get relevant data
        // Get density thermodynamic function
        ttf densityFunc = monitor->eos->GetThermodynamicFunction(tp::Density, monitor->GetSolver()->GetSubDomain().GetFields(fLoc::SOL));

        // Get cell range (just keep data in this struct and read out as necessary).
        Range cellRange;
        monitor->GetSolver()->GetCellRange(cellRange);

        // Extract the field vector global array
        const PetscScalar* fieldDat;
        ierr = VecGetArrayRead(vec, &fieldDat);
        CHKERRQ(ierr);

        // Extract the solution global array
        const PetscScalar* solDat;
        ierr = VecGetArrayRead(solVec, &solDat);
        CHKERRQ(ierr);

        // Get the array of turbulent flow data
        PetscScalar* turbDat;
        ierr = VecGetArray(monitor->turbVec, &turbDat);
        CHKERRQ(ierr);

        //! Compute measures
        // Get the timestep from the TS
        PetscReal dt;
        ierr = TSGetTimeStep(ts, &dt);
        CHKERRQ(ierr);

        // Instantiate running sums needed for RMS, MRMS
        std::vector<double> sum(field.numberComponents, 0.0);
        std::vector<double> sum2(field.numberComponents, 0.0);
        std::vector<double> mSum(field.numberComponents, 0.0);
        std::vector<double> mSum2(field.numberComponents, 0.0);
        double densSum = 0.0;

        // REMOVE THIS!!!
        VecView(monitor->turbVec, PETSC_VIEWER_STDOUT_WORLD);

        for (int c = cellRange.start; c < cellRange.end; c++) {
            const PetscScalar* fieldPt;
            const PetscScalar* solPt;
            PetscScalar* turbPt;

            // Get field point data
            ierr = DMPlexPointLocalRead(subDM, cellRange.points[c], fieldDat, &fieldPt);
            CHKERRQ(ierr);

            // Get solution point data
            ierr = DMPlexPointLocalRead(solDM, cellRange.points[c], solDat, &solPt);
            CHKERRQ(ierr);

            // Get read/write access to point in turbulent flow array
            ierr = DMPlexPointLocalRef(monitor->turbDM, cellRange.points[c], turbDat, &turbPt);
            CHKERRQ(ierr);

            // Get the density data from solution point data
            PetscReal densLoc;
            densityFunc.function(solPt, &densLoc, densityFunc.context.get());

            // Perform actual calculations now
            // Add density*dt at this point to the running sum
            turbPt[FieldOffset::dSum] += densLoc * dt;

            // Iterate through the components of the field values at this point
            for (int j = 0; j < field.numberComponents; j++) {
                // Multiply field value by density*dt
                turbPt[FieldOffset::dMult + j] += densLoc * fieldPt[j] * dt;

                // Add field value to spatial running sum
                sum[j] += fieldPt[j];

                // Square field value
                sum2[j] += PetscPowReal(fieldPt[j], 2);

                // Add density multiplied field value to running sum
                mSum[j] += fieldPt[j] * densLoc;

                // Multiply sqare field value by density
                mSum2[j] += sum2[j] * densLoc;
            }

            // Add density to spatial running sum
            densSum += densLoc;
        }

        // Free the turbulent flow array
        ierr = VecRestoreArray(monitor->turbVec, &turbDat);
        CHKERRQ(ierr);

        // REMOVE THIS!!!
        VecView(monitor->turbVec, PETSC_VIEWER_STDOUT_WORLD);

        // Calculate RMS and mRMS
        std::vector<double> rms(field.numberComponents, 0.0);
        std::vector<double> mRms(field.numberComponents, 0.0);

        for (int c = 0; c < field.numberComponents; c++) {
            rms[c] = PetscSqrtReal(sum2[c] / (cellRange.end - cellRange.start + tiny) - PetscPowReal(sum[c] / (cellRange.end - cellRange.start + tiny), 2));
            mRms[c] = PetscSqrtReal(mSum2[c] / (densSum + tiny) - PetscPowReal(mSum[c] / (densSum + tiny), 2));
        }

        // Output the RMS and mRMS data to the log
        monitor->log->Printf("Turb Flow Stats\n");
        monitor->log->Print("RMS", rms.size(), &rms[0], "%3.2e");
        monitor->log->Print("\nMRMS", mRms.size(), &mRms[0], "%3.2e");
        monitor->log->Printf("\n");

        PetscFunctionReturn(0);
    }
}

void ablate::monitors::TurbFlowStats::AddField(DM &dm, const char* nameField, const char* nameRegion, PetscInt numComp) {
    PetscFV fvm;
    PetscFVCreate(PetscObjectComm(PetscObject(dm)), &fvm) >>checkError;
    PetscObjectSetName((PetscObject)fvm, nameField) >>checkError;
    PetscFVSetFromOptions(fvm) >>checkError;
    PetscFVSetNumComponents(fvm, numComp) >>checkError;

    DMLabel thisRegion;
    DMGetLabel(dm, nameRegion, &thisRegion) >>checkError;
    DMAddField(dm, thisRegion, (PetscObject)fvm) >>checkError;
    PetscFVDestroy(&fvm);
}

void ablate::monitors::TurbFlowStats::Register(std::shared_ptr<ablate::solver::Solver> solverIn) {

    //Copy over the solver
    ablate::monitors::Monitor::Register(solverIn);

    //Copy the master DM over, set fields, and initialize vector values
    DM coordDM;
    DMGetCoordinateDM(this->GetSolver()->GetSubDomain().GetDM(), &coordDM) >>checkError;

    DMClone(this->GetSolver()->GetSubDomain().GetDM(), &turbDM);
    DMSetCoordinateDM(turbDM, coordDM) >>checkError;

    PetscInt numComp = this->GetSolver()->GetSubDomain().GetField(fieldName).numberComponents;
    //MUST include densitySum first, then densityMult
    std::string densitySum = "densitySum";
    AddField(turbDM, densitySum.c_str(), this->GetSolver()->GetRegion()->GetName().c_str(), 1);
    std::string densityMult = "densityMult";
    AddField(turbDM, densityMult.c_str(), this->GetSolver()->GetRegion()->GetName().c_str(), numComp);

    DMCreateGlobalVector(turbDM, &turbVec);
    VecSet(turbVec, 0.0);
}


#include <registrar.hpp>
REGISTER(ablate::monitors::Monitor, ablate::monitors::TurbFlowStats, "Computes turbulent flow statistics",
         ARG(std::string, "field", "The name of the field"),
         ARG(ablate::eos::EOS, "eos", "The equation of state"),
         OPT(ablate::monitors::logs::Log, "log", "Where the data will be sent (default is stdout)"),
         OPT(ablate::io::interval::Interval, "interval", "The monitor output interval"));
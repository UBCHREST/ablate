#include "fieldErrorMonitor.hpp"
#include <monitors/logs/stdOut.hpp>
#include "mathFunctions/mathFunction.hpp"

ablate::monitors::FieldErrorMonitor::FieldErrorMonitor(std::shared_ptr<logs::Log> logIn) : log(logIn ? logIn : std::make_shared<logs::StdOut>()) {}

PetscErrorCode ablate::monitors::FieldErrorMonitor::MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    FieldErrorMonitor *monitor = (FieldErrorMonitor *)ctx;
    // if this is the first time step init the log
    if (!monitor->log->Initialized()) {
        monitor->log->Initialize(monitor->GetSolver()->GetSubDomain().GetComm());
    }

    DM dm = monitor->GetSolver()->GetSubDomain().GetDM();
    PetscDS ds = monitor->GetSolver()->GetSubDomain().GetDiscreteSystem();

    // Get the number of fields
    PetscInt numberOfFields;
    ierr = DMGetNumFields(dm, &numberOfFields);
    CHKERRQ(ierr);

    // Get the exact funcs and contx
    std::vector<ablate::mathFunctions::PetscFunction> exactFuncs(numberOfFields, nullptr);
    std::vector<void *> ctxs(numberOfFields, nullptr);

    // Get the exact solution for this ds
    for (const auto &field : monitor->GetSolver()->GetSubDomain().GetFields()) {
        // Determine the solution location
        auto solId = monitor->GetSolver()->GetSubDomain().GetField(field.name);

        ierr = PetscDSGetExactSolution(ds, field.subId, &exactFuncs[solId.id], &ctxs[solId.id]);
        CHKERRQ(ierr);
    }

    // Store the errors
    std::vector<PetscReal> ferrors(numberOfFields);
    ierr = DMComputeL2FieldDiff(dm, crtime, &exactFuncs[0], &ctxs[0], u, &ferrors[0]);
    CHKERRQ(ierr);

    monitor->log->Printf("Timestep: %04d time = %-8.4g \t ", (int)step, (double)crtime);
    monitor->log->Print("L_2 Error", ferrors, "%2.3g");
    monitor->log->Print("\n");
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::FieldErrorMonitor, "Computes and reports the error every time step",
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
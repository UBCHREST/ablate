#include "fieldErrorMonitor.hpp"
#include <monitors/logs/stdOut.hpp>
#include "mathFunctions/mathFunction.hpp"

ablate::monitors::FieldErrorMonitor::FieldErrorMonitor(std::shared_ptr<logs::Log> logIn) : log(logIn ? logIn : std::make_shared<logs::StdOut>()) {}

PetscErrorCode ablate::monitors::FieldErrorMonitor::MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    DM dm;
    PetscDS ds;

    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    // Get the number of fields
    PetscInt numberOfFields;
    ierr = PetscDSGetNumFields(ds, &numberOfFields);
    CHKERRQ(ierr);

    FieldErrorMonitor *monitor = (FieldErrorMonitor *)ctx;
    // if this is the first time step init the log
    if (step == 0) {
        monitor->log->Initialize(PetscObjectComm((PetscObject)dm));
    }

    // Get the exact funcs and contx
    std::vector<ablate::mathFunctions::PetscFunction> exactFuncs(numberOfFields);
    std::vector<void *> ctxs(numberOfFields);
    for (auto f = 0; f < numberOfFields; ++f) {
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);
        CHKERRQ(ierr);
        if (!exactFuncs[f]) {
            SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_LIB, "The exact solution has not set");
        }
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

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::FieldErrorMonitor, "Computes and reports the error every time step",
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));
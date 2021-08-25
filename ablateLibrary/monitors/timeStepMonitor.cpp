#include "timeStepMonitor.hpp"
#include <monitors/logs/stdOut.hpp>

ablate::monitors::TimeStepMonitor::TimeStepMonitor(std::shared_ptr<logs::Log> logIn) : log(logIn ? logIn : std::make_shared<logs::StdOut>()) {}

PetscErrorCode ablate::monitors::TimeStepMonitor::MonitorTimeStep(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    PetscReal dt;
    ierr = TSGetTimeStep(ts, &dt);
    CHKERRQ(ierr);

    TimeStepMonitor* monitor = (TimeStepMonitor*)ctx;
    // if this is the first time step init the log
    if (!monitor->log->Initialized()) {
        monitor->log->Initialize(PetscObjectComm((PetscObject)ts));
    }

    monitor->log->Printf("Timestep: %04d time = %-8.4g dt = %g\n", (int)step, (double)crtime, (double)dt);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::TimeStepMonitor, "Reports the current step, time, and dt", OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));

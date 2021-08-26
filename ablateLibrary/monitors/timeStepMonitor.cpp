#include "timeStepMonitor.hpp"
#include <monitors/logs/stdOut.hpp>

ablate::monitors::TimeStepMonitor::TimeStepMonitor(std::shared_ptr<logs::Log> logIn, int interval) : log(logIn ? logIn : std::make_shared<logs::StdOut>()), interval(interval) {}

PetscErrorCode ablate::monitors::TimeStepMonitor::MonitorTimeStep(TS ts, PetscInt steps, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    auto monitor = (ablate::monitors::TimeStepMonitor*)ctx;

    if (steps == 0 || monitor->interval == 0 || (steps % monitor->interval == 0)) {
        PetscReal dt;
        ierr = TSGetTimeStep(ts, &dt);
        CHKERRQ(ierr);

        // if this is the first time step init the log
        if (!monitor->log->Initialized()) {
            monitor->log->Initialize(PetscObjectComm((PetscObject)ts));
        }

        monitor->log->Printf("Timestep: %04d time = %-8.4g dt = %g\n", (int)steps, (double)crtime, (double)dt);
        CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::TimeStepMonitor, "Reports the current step, time, and dt", OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"),
         OPT(int, "interval", "output interval"));

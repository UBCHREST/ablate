#include "timeStepMonitor.hpp"
#include <io/interval/fixedInterval.hpp>
#include <monitors/logs/stdOut.hpp>

ablate::monitors::TimeStepMonitor::TimeStepMonitor(std::shared_ptr<logs::Log> logIn, std::shared_ptr<io::interval::Interval> interval)
    : log(logIn ? logIn : std::make_shared<logs::StdOut>()), interval(interval ? interval : std::make_shared<io::interval::FixedInterval>()) {}

PetscErrorCode ablate::monitors::TimeStepMonitor::MonitorTimeStep(TS ts, PetscInt steps, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    auto monitor = (ablate::monitors::TimeStepMonitor*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), steps, crtime)) {
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

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::TimeStepMonitor, "Reports the current step, time, and dt", OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"),
         OPT(ablate::io::interval::Interval, "interval", "report interval object, defaults to every"));

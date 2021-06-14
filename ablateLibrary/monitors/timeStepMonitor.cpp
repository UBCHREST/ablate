#include "timeStepMonitor.hpp"
PetscErrorCode ablate::monitors::TimeStepMonitor::MonitorTimeStep(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    PetscReal dt;
    ierr = TSGetTimeStep(ts, &dt);
    CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g dt = %g\n", (int)step, (double)crtime, (double)dt);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::Monitor, ablate::monitors::TimeStepMonitor, "Reports the current step, time, and dt");

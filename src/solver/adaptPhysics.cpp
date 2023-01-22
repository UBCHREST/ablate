#include "adaptPhysics.hpp"
#include "timeStepper.hpp"
PetscErrorCode ablate::solver::AdaptPhysics::TSAdaptChoose(TSAdapt adapt, TS ts, PetscReal currentDt, PetscInt *next_sc, PetscReal *nextDt, PetscBool *accept, PetscReal *wlte, PetscReal *wltea,
                                                           PetscReal *wlter) {
    PetscFunctionBeginUser;
    // Get the time stepper context
    ablate::solver::TimeStepper *timeStepper;
    PetscCall(TSGetApplicationContext(ts, &timeStepper));

    // Compute the physics based timeStep from the time stepper
    PetscReal physicsDt;
    PetscCall(timeStepper->ComputePhysicsTimeStep(&physicsDt));

    /* Determine whether the step is accepted of rejected */
    *accept = PETSC_TRUE;

    // Scale the physics dt by the safety factor
    physicsDt *= adapt->safety;

    // Determine if we are on the first timestep
    PetscInt step;
    PetscCall(TSGetStepNumber(ts, &step));

    /* The optimal new step based purely on CFL constraint for this step. */
    *nextDt = PetscClipInterval(physicsDt, adapt->dt_min, adapt->dt_max);

    *next_sc = 0;
    *wlte = -1;  /* Weighted local truncation error was not evaluated */
    *wltea = -1; /* Weighted absolute local truncation error was not evaluated */
    *wlter = -1; /* Weighted relative local truncation error was not evaluated */
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::AdaptPhysics::TSAdaptCreate(TSAdapt adapt) {
    PetscFunctionBegin;
    adapt->ops->choose = TSAdaptChoose;

    // set the default behavior to always accept
    PetscCall(TSAdaptSetAlwaysAccept(adapt, PETSC_TRUE));

    PetscFunctionReturn(0);
}

void ablate::solver::AdaptPhysics::Register() {
    // Register the create function with petsc
    TSAdaptRegister(name, TSAdaptCreate) >> ablate::utilities::PetscUtilities::checkError;

    // Register the initializer function with ablate time stepper
    ablate::solver::TimeStepper::RegisterAdaptInitializer(std::string(name), AdaptInitializer);
}

void ablate::solver::AdaptPhysics::AdaptInitializer(TS ts, TSAdapt adapt) {
    // Get the time stepper context
    ablate::solver::TimeStepper *timeStepper;
    TSGetApplicationContext(ts, &timeStepper) >> ablate::utilities::PetscUtilities::checkError;

    // Compute the physics based timeStep from the time stepper
    PetscReal physicsDt;
    timeStepper->ComputePhysicsTimeStep(&physicsDt) >> ablate::utilities::PetscUtilities::checkError;

    // Scale the physics dt by the safety factor
    physicsDt *= adapt->safety;

    /* The optimal new step based purely on CFL constraint for this step. */
    physicsDt = PetscClipInterval(physicsDt, adapt->dt_min, adapt->dt_max);

    // Get the current dt
    PetscReal currentDt;
    TSGetTimeStep(ts, &currentDt) >> ablate::utilities::PetscUtilities::checkError;

    // only shrink the dt if it is smaller the proposed current dt
    if (physicsDt < currentDt) {
        TSSetTimeStep(ts, physicsDt) >> ablate::utilities::PetscUtilities::checkError;
    }
}

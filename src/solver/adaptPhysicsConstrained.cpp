#include "adaptPhysicsConstrained.hpp"
#include "timeStepper.hpp"

/**
 * This is a slightly modified version of TSAdaptChoose_Basic.  Ideally this would have called the the petsc implementation but it is a static private implementation. This is a small
 * amount of code so it was duplicated here.
 * @param adapt
 * @param ts
 * @param currentDt
 * @param next_sc
 * @param nextDt
 * @param accept
 * @param wlte
 * @param wltea
 * @param wlter
 * @return
 */
PetscErrorCode ablate::solver::AdaptPhysicsConstrained::TSAdaptChoose(TSAdapt adapt, TS ts, PetscReal h, PetscInt *next_sc, PetscReal *next_h, PetscBool *accept, PetscReal *wlte, PetscReal *wltea,
                                                           PetscReal *wlter) {
    Vec       Y;
    DM        dm;
    PetscInt  order = PETSC_DECIDE;
    PetscReal enorm = -1;
    PetscReal enorma, enormr;
    PetscReal safety = adapt->safety;
    PetscReal hfac_lte, h_lte;

    PetscFunctionBeginUser;
    *next_sc = 0;  /* Reuse the same order scheme */
    *wltea   = -1; /* Weighted absolute local truncation error is not used */
    *wlter   = -1; /* Weighted relative local truncation error is not used */

    if (ts->ops->evaluatewlte) {
        PetscCall(TSEvaluateWLTE(ts, adapt->wnormtype, &order, &enorm));
        PetscCheck(enorm < 0 || order >= 1, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_OUTOFRANGE, "Computed error order %" PetscInt_FMT " must be positive", order);
    } else if (ts->ops->evaluatestep) {
        PetscCheck(adapt->candidates.n >= 1, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_WRONGSTATE, "No candidate has been registered");
        PetscCheck(adapt->candidates.inuse_set, PetscObjectComm((PetscObject)adapt), PETSC_ERR_ARG_WRONGSTATE, "The current in-use scheme is not among the %" PetscInt_FMT " candidates", adapt->candidates.n);
        order = adapt->candidates.order[0];
        PetscCall(TSGetDM(ts, &dm));
        PetscCall(DMGetGlobalVector(dm, &Y));
        PetscCall(TSEvaluateStep(ts, order - 1, Y, NULL));
        PetscCall(TSErrorWeightedNorm(ts, ts->vec_sol, Y, adapt->wnormtype, &enorm, &enorma, &enormr));
        PetscCall(DMRestoreGlobalVector(dm, &Y));
    }

    if (enorm < 0) {
        *accept = PETSC_TRUE;
        *next_h = h;  /* Reuse the old step */
        *wlte   = -1; /* Weighted local truncation error was not evaluated */
        PetscFunctionReturn(0);
    }

    /* Determine whether the step is accepted of rejected */
    if (enorm > 1) {
        if (!*accept) safety *= adapt->reject_safety; /* The last attempt also failed, shorten more aggressively */
        if (h < (1 + PETSC_SQRT_MACHINE_EPSILON) * adapt->dt_min) {
            PetscCall(PetscInfo(adapt, "Estimated scaled local truncation error %g, accepting because step size %g is at minimum\n", (double)enorm, (double)h));
            *accept = PETSC_TRUE;
        } else if (adapt->always_accept) {
            PetscCall(PetscInfo(adapt, "Estimated scaled local truncation error %g, accepting step of size %g because always_accept is set\n", (double)enorm, (double)h));
            *accept = PETSC_TRUE;
        } else {
            PetscCall(PetscInfo(adapt, "Estimated scaled local truncation error %g, rejecting step of size %g\n", (double)enorm, (double)h));
            *accept = PETSC_FALSE;
        }
    } else {
        PetscCall(PetscInfo(adapt, "Estimated scaled local truncation error %g, accepting step of size %g\n", (double)enorm, (double)h));
        *accept = PETSC_TRUE;
    }

    /* The optimal new step based purely on local truncation error for this step. */
    if (enorm > 0) hfac_lte = safety * PetscPowReal(enorm, ((PetscReal)-1) / order);
    else hfac_lte = safety * PETSC_INFINITY;
    if (adapt->timestepjustdecreased) {
        hfac_lte = PetscMin(hfac_lte, 1.0);
        adapt->timestepjustdecreased--;
    }
    h_lte = h * PetscClipInterval(hfac_lte, adapt->clip[0], adapt->clip[1]);

    *next_h = PetscClipInterval(h_lte, adapt->dt_min, adapt->dt_max);

    {// This is the ablate specific code that clips based upon the physics dt
       // Get the time stepper context
         ablate::solver::TimeStepper *timeStepper;
         PetscCall(TSGetApplicationContext(ts, &timeStepper));

         // Compute the physics based timeStep from the time stepper
         PetscReal physicsDt;
         PetscCall(timeStepper->ComputePhysicsTimeStep(&physicsDt));

         // Scale the physics dt by the safety factor
         physicsDt *= adapt->safety;

         if(physicsDt < *next_h){
            PetscCall(PetscInfo(adapt, "Limiting adaptive time step %g, to physics maximum dt %g\n", (double)*next_h, (double)physicsDt));
            *next_h = physicsDt;
         }
    }

    *wlte   = enorm;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::AdaptPhysicsConstrained::TSAdaptCreate(TSAdapt adapt) {
    PetscFunctionBegin;
    adapt->ops->choose = TSAdaptChoose;

    PetscFunctionReturn(0);
}

void ablate::solver::AdaptPhysicsConstrained::Register() {
    // Register the create function with petsc
    TSAdaptRegister(name, TSAdaptCreate) >> ablate::utilities::PetscUtilities::checkError;

    // Register the initializer function with ablate time stepper
    ablate::solver::TimeStepper::RegisterAdaptInitializer(std::string(name), AdaptInitializer);
}

void ablate::solver::AdaptPhysicsConstrained::AdaptInitializer(TS ts, TSAdapt adapt) {
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

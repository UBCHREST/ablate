#include "testingAuxFieldUpdater.hpp"

PetscErrorCode tests::ablateCore::support::TestingAuxFieldUpdater::UpdateSourceTerms(TS ts, ablate::flow::Flow& flow) {
    PetscFunctionBegin;
    // extract the flow data from ts
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    CHKERRQ(ierr);

    // get the time at the end of the time step
    PetscReal time;
    ierr = TSGetTime(ts, &time);
    CHKERRQ(ierr);
    PetscReal dt;
    ierr = TSGetTimeStep(ts, &dt);
    CHKERRQ(ierr);

    // Update the source terms
    ierr = DMProjectFunctionLocal(flow.GetAuxDM(), time + dt, &(funcs[0]), &(ctxs[0]), INSERT_ALL_VALUES, flow.GetAuxField());
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

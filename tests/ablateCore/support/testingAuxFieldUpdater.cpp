#include "testingAuxFieldUpdater.hpp"
#include "flow.h"

PetscErrorCode tests::ablateCore::support::TestingAuxFieldUpdater::UpdateSourceTerms(TS ts, void *ctx) {
    PetscFunctionBegin;
    auto sourceTermUpdater = (TestingAuxFieldUpdater *)ctx;

    // extract the flow data from ts
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    FlowData flowData;
    ierr = DMGetApplicationContext(dm, &flowData);
    CHKERRQ(ierr);

    // get the time at the end of the time step
    PetscReal time;
    ierr = TSGetTime(ts, &time);
    CHKERRQ(ierr);
    PetscReal dt;
    ierr = TSGetTimeStep(ts, &dt);
    CHKERRQ(ierr);

    // Update the source terms
    ierr = DMProjectFunctionLocal(flowData->auxDm, time + dt, &(sourceTermUpdater->funcs[0]), &(sourceTermUpdater->ctxs[0]), INSERT_ALL_VALUES, flowData->auxField);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

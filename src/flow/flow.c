#include "flow.h"
#include "incompressibleFlow.h"
#include "lowMachFlow.h"

PetscErrorCode FlowCreate(Flow* flow, FlowType type, DM dm) {
    *flow = malloc(sizeof(struct _Flow));

    // associate the dm
    (*flow)->dm = dm;
    (*flow)->data = NULL;

    PetscErrorCode ierr = DMSetApplicationContext(dm, *flow);CHKERRQ(ierr);

    // setup the parameters
    ierr = SetupFlowParameters(&(*flow)->parameters);CHKERRQ(ierr);

    if (!type) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The flow type must be specified");
    } else if (strcmp(type, FLOWLOWMACH) == 0) {
        return LowMachFlowCreate(*flow);
    } else if (strcmp(type, FLOWINCOMPRESSIBLE) == 0) {
        return IncompressibleFlowCreate(*flow);
    }
    { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown flow type"); }
}

PETSC_EXTERN PetscErrorCode FlowDestroy(Flow* flow) {
    PetscErrorCode ierr = (*flow)->destroy(*flow);CHKERRQ(ierr);
    ierr = PetscBagDestroy(&(*flow)->parameters);CHKERRQ(ierr);
    free(*flow);
    flow = NULL;
    return 0;
}

PETSC_EXTERN PetscErrorCode FlowSetupDiscretization(Flow flow) { return flow->setupDiscretization(flow); }

PETSC_EXTERN PetscErrorCode FlowInitialization(Flow flow, DM dm, Vec u) {
    if (flow->completeFlowInitialization) {
        return flow->completeFlowInitialization(dm, u);
    } else {
        return 0;
    }
}

PETSC_EXTERN PetscErrorCode FlowStartProblemSetup(Flow flow) { return flow->startProblemSetup(flow); }
PETSC_EXTERN PetscErrorCode FlowCompleteProblemSetup(Flow flow, TS ts) { return flow->completeProblemSetup(flow, ts); }
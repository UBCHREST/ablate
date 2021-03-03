#include "flow.h"

PetscErrorCode FlowCreate(FlowData* flow, PetscInt numberOfFields) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    *flow = malloc(sizeof(struct _FlowData));

    // initialize all fields
    (*flow)->dm = NULL;
    (*flow)->data = NULL;
    (*flow)->flowSolution = NULL;

    PetscFunctionReturn(0);
}

PetscErrorCode FlowDestroy(FlowData* flow) {
    PetscFunctionBeginUser;
    free(*flow);
    flow = NULL;
    PetscFunctionReturn(0);
}
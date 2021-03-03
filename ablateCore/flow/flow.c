#include "flow.h"

PetscErrorCode FlowCreate(FlowData* flow) {
    PetscFunctionBeginUser;
    *flow = malloc(sizeof(struct _FlowData));

    // initialize all fields
    (*flow)->dm = NULL;
    (*flow)->data = NULL;
    (*flow)->flowField = NULL;

    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterFields(FlowData flow, PetscInt numberFlowFields, const char* const* flowFields) {
    PetscFunctionBeginUser;
    flow->numberFlowFields = numberFlowFields;

    PetscErrorCode ierr = PetscStrNArrayallocpy(numberFlowFields, flowFields, (char***)&(flow->flowFieldNames));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode FlowDestroy(FlowData* flow) {
    PetscFunctionBeginUser;
    if((*flow)->flowFieldNames){
        PetscErrorCode ierr = PetscStrNArrayDestroy((*flow)->numberFlowFields,(char***)&((*flow)->flowFieldNames));CHKERRQ(ierr);
    }
    if((*flow)->flowField){
        PetscErrorCode ierr = VecDestroy(&((*flow)->flowField));CHKERRQ(ierr);
    }
    free(*flow);
    flow = NULL;
    PetscFunctionReturn(0);
}
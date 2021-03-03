#ifndef ABLATELIBRARY_FLOW_H
#define ABLATELIBRARY_FLOW_H

#include <petsc.h>

struct _FlowData {
    DM dm;               /* flow domain */
    Vec flowField;  /* The solution to the flow */

    void* data; /* implementation-specific data */

    PetscInt numberFlowFields;
    const char *const * flowFieldNames;
};

typedef struct _FlowData* FlowData;

PETSC_EXTERN PetscErrorCode FlowCreate(FlowData* flow);
PetscErrorCode FlowRegisterFields(FlowData flow, PetscInt numberFlowFields, const char *const *flowFields);

PETSC_EXTERN PetscErrorCode FlowDestroy(FlowData* flow);

#endif  // ABLATELIBRARY_FLOW_H

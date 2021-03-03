#ifndef ABLATELIBRARY_FLOW_H
#define ABLATELIBRARY_FLOW_H

#include <petsc.h>

struct _FlowData {
    DM dm;               /* flow domain */
    Vec flowSolution;  /* The solution to the flow */

    void* data; /* implementation-specific data */

};

typedef struct _FlowData* FlowData;

PetscErrorCode FlowCreate(FlowData* flow, PetscInt numberOfFields);
PetscErrorCode FlowDestroy(FlowData* flow);

#endif  // ABLATELIBRARY_FLOW_H

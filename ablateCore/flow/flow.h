#ifndef ABLATELIBRARY_FLOW_H
#define ABLATELIBRARY_FLOW_H

#include <petsc.h>

typedef struct {
    const char* fieldName;
    const char* fieldPrefix;
    PetscInt components;
} FlowFieldDescriptor;


struct _FlowData {
    DM dm;               /* flow domain */
    Vec flowField;  /* The solution to the flow */

    void* data; /* implementation-specific data */

    /** store the fields on the dm swarm **/
    PetscInt numberFlowFields;
    FlowFieldDescriptor *fieldDescriptors;
};

typedef struct _FlowData* FlowData;

PETSC_EXTERN PetscErrorCode FlowCreate(FlowData* flow);
PetscErrorCode FlowRegisterFields(FlowData flow, const char fieldName[],const char fieldPrefix[], PetscInt components);

PETSC_EXTERN PetscErrorCode FlowDestroy(FlowData* flow);

#endif  // ABLATELIBRARY_FLOW_H

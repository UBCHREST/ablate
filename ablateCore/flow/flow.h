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
    DM auxDm;               /* holds non solution vector fields */

    Vec flowField;  /* The solution to the flow */
    Vec auxField;  /* The aux field to the flow */

    void* data; /* implementation-specific data */

    /** store the fields on the dm**/
    PetscInt numberFlowFields;
    FlowFieldDescriptor * flowFieldDescriptors;

    /** store the aux fields **/
    PetscInt numberAuxFields;
    FlowFieldDescriptor * auxFieldDescriptors;
};

typedef struct _FlowData* FlowData;

PETSC_EXTERN PetscErrorCode FlowCreate(FlowData* flow);
PetscErrorCode FlowRegisterField(FlowData flow, const char* fieldName, const char* fieldPrefix, PetscInt components);
PetscErrorCode FlowRegisterAuxField(FlowData flow, const char* fieldName, const char* fieldPrefix, PetscInt components);
PetscErrorCode FlowCompleteProblemSetup(FlowData flow, TS ts);
PetscErrorCode FlowFinalizeRegisterFields(FlowData flow);
PETSC_EXTERN PetscErrorCode FlowDestroy(FlowData* flow);

#endif  // ABLATELIBRARY_FLOW_H

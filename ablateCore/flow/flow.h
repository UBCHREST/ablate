#ifndef ABLATELIBRARY_FLOW_H
#define ABLATELIBRARY_FLOW_H

#include <petsc.h>

enum FieldType {FE, FV};

typedef struct {
    const char* fieldName;
    const char* fieldPrefix;
    PetscInt components;
    enum FieldType fieldType;
} FlowFieldDescriptor;

typedef struct {
    void* context;
    PetscErrorCode (*updateFunction)(TS ts, void* context);
} FlowUpdateFunction;

struct _FlowData {
    DM dm;    /* flow domain */
    DM auxDm; /* holds non solution vector fields */

    Vec flowField; /* The solution to the flow */
    Vec auxField;  /* The aux field to the flow */

    void* data; /* implementation-specific data */

    /** store the fields on the dm**/
    PetscInt numberFlowFields;
    FlowFieldDescriptor* flowFieldDescriptors;

    /** store the aux fields **/
    PetscInt numberAuxFields;
    FlowFieldDescriptor* auxFieldDescriptors;

    /** store a list of preStepFunctions and postStepFunctions **/
    PetscInt numberPreStepFunctions;
    FlowUpdateFunction* preStepFunctions;
    PetscInt numberPostStepFunctions;
    FlowUpdateFunction* postStepFunctions;
};

typedef struct _FlowData* FlowData;

PETSC_EXTERN PetscErrorCode FlowCreate(FlowData* flow);
PetscErrorCode FlowRegisterField(FlowData flow, const char* fieldName, const char* fieldPrefix, PetscInt components, enum FieldType fieldType);
PetscErrorCode FlowRegisterAuxField(FlowData flow, const char* fieldName, const char* fieldPrefix, PetscInt components, enum FieldType fieldType);
PetscErrorCode FlowCompleteProblemSetup(FlowData flow, TS ts);
PetscErrorCode FlowFinalizeRegisterFields(FlowData flow);
PETSC_EXTERN PetscErrorCode FlowRegisterPreStep(FlowData flowData, PetscErrorCode (*updateFunction)(TS ts, void* context), void* context);
PETSC_EXTERN PetscErrorCode FlowRegisterPostStep(FlowData flowData, PetscErrorCode (*updateFunction)(TS ts, void* context), void* context);
PETSC_EXTERN PetscErrorCode FlowView(FlowData flowData,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode FlowViewFromOptions(FlowData flowData, char *title);

PETSC_EXTERN PetscErrorCode FlowDestroy(FlowData* flow);

#endif  // ABLATELIBRARY_FLOW_H

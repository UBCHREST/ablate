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
    // the specific type of the flow
    char* type;

    DM dm;    /* flow domain */
    DM auxDm; /* holds non solution vector fields */

    Vec flowField; /* The solution to the flow */
    Vec auxField;  /* The aux field to the flow */

    void* data; /* implementation-specific data */

    // the options databased used to setup the options.  If not set, the default (NULL) options is used
    PetscOptions options;

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

    /** implementation-specific methods **/
    PetscErrorCode (*flowSetupDiscretization)(struct _FlowData*, DM* dm);
    PetscErrorCode (*flowStartProblemSetup)(struct _FlowData*);
    PetscErrorCode (*flowCompleteProblemSetup)(struct _FlowData*, TS ts);
    PetscErrorCode (*flowCompleteFlowInitialization)(DM dm, Vec u);
    PetscErrorCode (*flowDestroy)(struct _FlowData*);
};

typedef struct _FlowData* FlowData;

typedef PetscErrorCode (*FlowSetupFunction)(FlowData flow);

PETSC_EXTERN PetscErrorCode FlowRegister(const char*, const FlowSetupFunction);

// Flow object management methods
PETSC_EXTERN PetscErrorCode FlowCreate(FlowData* flow);
PETSC_EXTERN PetscErrorCode FlowSetType(FlowData flow, const char*);
PETSC_EXTERN PetscErrorCode FlowSetOptions(FlowData flow, PetscOptions options);
PETSC_EXTERN PetscErrorCode FlowSetFromOptions(FlowData flow);

// Concrete methods for flow
PETSC_EXTERN PetscErrorCode FlowRegisterPreStep(FlowData flowData, PetscErrorCode (*updateFunction)(TS ts, void* context), void* context);
PETSC_EXTERN PetscErrorCode FlowRegisterPostStep(FlowData flowData, PetscErrorCode (*updateFunction)(TS ts, void* context), void* context);
PETSC_EXTERN PetscErrorCode FlowView(FlowData flowData,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode FlowViewFromOptions(FlowData flowData, char *title);

// Abstract methods
PETSC_EXTERN PetscErrorCode FlowSetupDiscretization(FlowData flowData, DM* dm);
PETSC_EXTERN PetscErrorCode FlowStartProblemSetup(FlowData flowData);
PETSC_EXTERN PetscErrorCode FlowCompleteProblemSetup(FlowData flow, TS ts);
PETSC_EXTERN PetscErrorCode FlowCompleteFlowInitialization(FlowData flow, DM, Vec);
PETSC_EXTERN PetscErrorCode FlowDestroy(FlowData* flow);

// Methods that should be used from flows
PetscErrorCode FlowRegisterField(FlowData flow, const char* fieldName, const char* fieldPrefix, PetscInt components, enum FieldType fieldType);
PetscErrorCode FlowRegisterAuxField(FlowData flow, const char* fieldName, const char* fieldPrefix, PetscInt components, enum FieldType fieldType);
PetscErrorCode FlowFinalizeRegisterFields(FlowData flow);

#endif  // ABLATELIBRARY_FLOW_H

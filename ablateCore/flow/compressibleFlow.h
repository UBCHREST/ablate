#if !defined(compressibleFlow_h)
#define compressibleFlow_h
#include <petsc.h>
#include "flow.h"

typedef enum {RHO, RHOU, RHOE, TOTAL_COMPRESSIBLE_FLOW_FIELDS} CompressibleFlowFields;

typedef struct {
    PetscReal cfl;
    PetscReal gamma;
} EulerFlowParameters;



// Define the functions to setup the incompressible flow
PETSC_EXTERN PetscErrorCode CompressibleFlow_SetupFlowParameters(FlowData flowData, EulerFlowParameters* eulerFlowParameters);
PETSC_EXTERN PetscErrorCode CompressibleFlow_SetupDiscretization(FlowData flowData, DM dm);
PETSC_EXTERN PetscErrorCode CompressibleFlow_StartProblemSetup(FlowData flowData);
PETSC_EXTERN PetscErrorCode CompressibleFlow_CompleteProblemSetup(FlowData flowData, TS ts);
//PETSC_EXTERN PetscErrorCode CompressibleFlow_CompleteFlowInitialization(DM dm, Vec u);
//PETSC_EXTERN PetscErrorCode CompressibleFlow_EnableAuxFields(FlowData flowData);

#endif
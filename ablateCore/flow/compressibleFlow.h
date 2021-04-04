#if !defined(compressibleFlow_h)
#define compressibleFlow_h
#include <petsc.h>
#include "flow.h"

typedef enum {RHO, RHOU, RHOE, TOTAL_COMPRESSIBLE_FLOW_FIELDS} CompressibleFlowFields;

typedef struct {
    PetscReal cfl;
    PetscReal gamma;
} EulerFlowParameters;

typedef struct {
    PetscReal gamma;
    PetscReal rhoL;
    PetscReal rhoR;
    PetscReal uL;
    PetscReal uR;
    PetscReal pL;
    PetscReal pR;
    PetscReal maxTime;
    PetscReal length;
} InitialConditions;

typedef struct {
    PetscReal rho;
    PetscReal rhoU[2];//Hardcode for 2 for now
    PetscReal rhoE;
} EulerNode;

typedef struct {
    PetscReal pstar,ustar,rhostarL,astarL,SL,SHL,STL,rhostarR,astarR,SR,SHR,STR,gamm1,gamp1;
} StarState;


// Define the functions to setup the incompressible flow
PETSC_EXTERN PetscErrorCode CompressibleFlow_SetupFlowParameters(FlowData flowData, const EulerFlowParameters* eulerFlowParameters);
PETSC_EXTERN PetscErrorCode CompressibleFlow_SetupDiscretization(FlowData flowData, DM dm);
PETSC_EXTERN PetscErrorCode CompressibleFlow_StartProblemSetup(FlowData flowData);
PETSC_EXTERN PetscErrorCode CompressibleFlow_CompleteProblemSetup(FlowData flowData, TS ts);
//PETSC_EXTERN PetscErrorCode CompressibleFlow_CompleteFlowInitialization(DM dm, Vec u);
//PETSC_EXTERN PetscErrorCode CompressibleFlow_EnableAuxFields(FlowData flowData);

void SetExactSolutionAtPoint(PetscInt dim, PetscReal xDt, const InitialConditions* setup, const StarState* starState, EulerNode* uu);
PetscErrorCode DetermineStarState(const InitialConditions* setup, StarState* starState);

#endif
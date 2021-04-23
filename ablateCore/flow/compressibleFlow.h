#if !defined(compressibleFlow_h)
#define compressibleFlow_h
#include <petsc.h>
#include "flow.h"
#include "fluxDifferencer.h"

typedef enum {RHO, RHOE, RHOU, RHOV, RHOW, TOTAL_COMPRESSIBLE_FLOW_COMPONENTS} CompressibleFlowComponents;
typedef enum { CFL, GAMMA, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS } CompressibleFlowParametersTypes;
PETSC_EXTERN const char *compressibleFlowParametersTypeNames[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS + 1];

typedef struct {
    PetscReal cfl;
    PetscReal gamma;
    FluxDifferencerFunction fluxDifferencer;
} EulerFlowData;

// Define the functions to setup the incompressible flow
PETSC_EXTERN PetscErrorCode CompressibleFlow_SetupDiscretization(FlowData flowData, DM dm);
PETSC_EXTERN PetscErrorCode CompressibleFlow_StartProblemSetup(FlowData flowData, PetscInt, PetscScalar []);
PETSC_EXTERN PetscErrorCode CompressibleFlow_CompleteProblemSetup(FlowData flowData, TS ts);

PETSC_EXTERN void CompressibleFlowComputeEulerFlux(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx);

#endif
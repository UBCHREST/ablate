#if !defined(compressibleFlow_h)
#define compressibleFlow_h
#include <petsc.h>
#include "flow.h"
#include "fluxDifferencer.h"
#include "eos.h"

typedef enum {RHO, RHOE, RHOU, RHOV, RHOW, TOTAL_COMPRESSIBLE_FLOW_COMPONENTS} CompressibleFlowComponents;
typedef enum {T, VEL, TOTAL_COMPRESSIBLE_AUX_COMPONENTS} CompressibleAuxComponents;

typedef PetscErrorCode (*FVAuxFieldUpdateFunction)(FlowData flowData, PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField);

struct _FlowData_CompressibleFlow{
    PetscReal cfl;
    PetscReal k;/*thermal conductivity*/
    PetscReal mu;/*dynamic viscosity*/
    FluxDifferencerFunction fluxDifferencer;
    PetscBool automaticTimeStepCalculator;
    EOSData eos;

    // functions to update each aux field
    FVAuxFieldUpdateFunction auxFieldUpdateFunctions[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
} ;

typedef struct _FlowData_CompressibleFlow* FlowData_CompressibleFlow;

PETSC_EXTERN PetscErrorCode CompressibleFlow_SetEOS(FlowData flowData, EOSData eosData);
PetscErrorCode FlowSetFromOptions_CompressibleFlow(FlowData flow);

PETSC_EXTERN void CompressibleFlowComputeEulerFlux(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx);
PETSC_EXTERN PetscErrorCode CompressibleFlowRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx);
PETSC_EXTERN PetscErrorCode CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVelL, const PetscReal * gradVelR, PetscReal* tau);

#endif
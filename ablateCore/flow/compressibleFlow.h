#if !defined(compressibleFlow_h)
#define compressibleFlow_h
#include <petsc.h>
#include "fluxDifferencer.h"

typedef enum {RHO, RHOE, RHOU, RHOV, RHOW, TOTAL_COMPRESSIBLE_FLOW_COMPONENTS} CompressibleFlowComponents;
typedef enum {T, VEL, TOTAL_COMPRESSIBLE_AUX_COMPONENTS} CompressibleAuxComponents;

struct _FlowData_CompressibleFlow{
    /*Courant Friedrichs Lewy*/
    PetscReal cfl;
    /* thermal conductivity*/
    PetscReal k;
    /* dynamic viscosity*/
    PetscReal mu;

    /* store method used for flux differencer */
    FluxDifferencerFunction fluxDifferencer;

    // EOS function calls
    PetscErrorCode (*decodeStateFunction)(const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx);
    void* decodeStateFunctionContext;
    PetscErrorCode (*computeTemperatureFunction)(const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, PetscReal* T, void* ctx);
    void* computeTemperatureContext;

    PetscBool automaticTimeStepCalculator;
} ;

typedef struct _FlowData_CompressibleFlow* FlowData_CompressibleFlow;

typedef PetscErrorCode (*FVAuxFieldUpdateFunction)(FlowData_CompressibleFlow flowData, PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField);

PETSC_EXTERN PetscErrorCode FVFlowUpdateAuxFieldsFV(DM dm, DM auxDM, PetscReal time, Vec locXVec, Vec locAuxField, PetscInt numberUpdateFunctions, FVAuxFieldUpdateFunction* updateFunctions, FlowData_CompressibleFlow data);

PETSC_EXTERN PetscErrorCode CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVelL, const PetscReal * gradVelR, PetscReal* tau);

// PreBuilt Flux functions
PETSC_EXTERN PetscErrorCode CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom* fg,
                                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[],
                                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                                             PetscScalar* flux, void* ctx);

PETSC_EXTERN PetscErrorCode CompressibleFlowEulerDiffusion(PetscInt dim, const PetscFVFaceGeom* fg,
                                                           const PetscInt uOff[], const PetscInt uOff_x[],  const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[],
                                                           const PetscInt aOff[], const PetscInt aOff_x[],  const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                                           PetscScalar* fL, void* ctx);

#endif
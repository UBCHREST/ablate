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

PETSC_EXTERN PetscErrorCode CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVelL, const PetscReal * gradVelR, PetscReal* tau);

// PreBuilt Flux functions

/**
 * This Computes the Flow Euler flow for rho, rhoE, and rhoVel.
 * u = {"euler"}
 * a = {}
 * ctx = FlowData_CompressibleFlow
 * @return
 */
PETSC_EXTERN PetscErrorCode CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom* fg,
                                                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[],
                                                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                                             PetscScalar* flux, void* ctx);

/**
 * This Computes the diffusion flux for euler rhoE, rhoVel
 * u = {"euler"}
 * a = {"temperature", "velocity"}
 * ctx = FlowData_CompressibleFlow
 * @return
 */
PETSC_EXTERN PetscErrorCode CompressibleFlowEulerDiffusion(PetscInt dim, const PetscFVFaceGeom* fg,
                                                           const PetscInt uOff[], const PetscInt uOff_x[],  const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[],
                                                           const PetscInt aOff[], const PetscInt aOff_x[],  const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                                           PetscScalar* fL, void* ctx);

#endif
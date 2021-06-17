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

typedef PetscErrorCode (*FVDiffusionFunction)(FlowData_CompressibleFlow flowData, PetscReal time, PetscInt dim, const PetscFVFaceGeom* fg, const PetscFVCellGeom* cgL, const PetscFVCellGeom* cgR,
                                              const PetscScalar* fieldL, const PetscScalar* fieldR,const PetscScalar* auxL, const PetscScalar* auxR,
                                              const PetscScalar** gradAuxL, const PetscScalar** gradAuxR, PetscScalar* fluxL, PetscScalar* fluxR);

PETSC_EXTERN PetscErrorCode FVFlowUpdateAuxFieldsFV(DM dm, DM auxDM, PetscReal time, Vec locXVec, Vec locAuxField, PetscInt numberUpdateFunctions, FVAuxFieldUpdateFunction* updateFunctions, FlowData_CompressibleFlow data);
PETSC_EXTERN PetscErrorCode CompressibleFlowDiffusionSourceRHSFunctionLocal(DM dm, DM auxDM, PetscReal time, Vec locXVec, Vec locAuxVec, Vec globFVec, FlowData_CompressibleFlow flowParameters, FVDiffusionFunction*);
PETSC_EXTERN void CompressibleFlowComputeEulerFlux(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx);
PETSC_EXTERN PetscErrorCode CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVelL, const PetscReal * gradVelR, PetscReal* tau);

PETSC_EXTERN PetscErrorCode CompressibleFlowEulerDiffusion(FlowData_CompressibleFlow flowParameters, PetscReal time, PetscInt dim, const PetscFVFaceGeom* fg, const PetscFVCellGeom* cgL, const PetscFVCellGeom* cgR,
                                              const PetscScalar* fieldL, const PetscScalar* fieldR,const PetscScalar* auxL, const PetscScalar* auxR,
                                              const PetscScalar** gradAuxL, const PetscScalar** gradAuxR, PetscScalar* fL, PetscScalar* fR);

#endif
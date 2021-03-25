#if !defined(incompressibleFlow_h)
#define incompressibleFlow_h
#include <petsc.h>
#include "flow.h"

// Define the test field number
#define VTEST 0
#define QTEST 1
#define WTEST 2

typedef enum { STROUHAL, REYNOLDS, PECLET, MU, K, CP, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS } IncompressibleFlowParametersTypes;
PETSC_EXTERN const char *incompressibleFlowParametersTypeNames[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS + 1];

typedef enum {VEL, PRES, TEMP, TOTAL_INCOMPRESSIBLE_FLOW_FIELDS} IncompressibleFlowFields;

typedef enum {MOM, MASS, ENERGY, TOTAL_INCOMPRESSIBLE_SOURCE_FIELDS} IncompressibleSourceFields;

typedef struct {
    PetscReal strouhal;
    PetscReal reynolds;
    PetscReal peclet;
    PetscReal mu; /* non-dimensional viscosity */
    PetscReal k;  /* non-dimensional thermal conductivity */
    PetscReal cp; /* non-dimensional specific heat capacity */
} IncompressibleFlowParameters;

PETSC_EXTERN PetscErrorCode IncompressibleFlow_PackParameters(IncompressibleFlowParameters *parameters, PetscScalar *constantArray);
PETSC_EXTERN PetscErrorCode IncompressibleFlow_ParametersFromPETScOptions(PetscBag *flowParametersBag);

// Define the functions to setup the incompressible flow
PETSC_EXTERN PetscErrorCode IncompressibleFlow_SetupDiscretization(FlowData flowData, DM dm);
PETSC_EXTERN PetscErrorCode IncompressibleFlow_StartProblemSetup(FlowData flowData, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode IncompressibleFlow_CompleteProblemSetup(FlowData flowData, TS ts);
PETSC_EXTERN PetscErrorCode IncompressibleFlow_CompleteFlowInitialization(DM dm, Vec u);
PETSC_EXTERN PetscErrorCode IncompressibleFlow_EnableAuxFields(FlowData flowData);

#endif
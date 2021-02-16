#if !defined(lowMachFlow_h)
#define lowMachFlow_h
#include <petsc.h>

// Define field id for velocity, pressure, temperature
#define VEL 0
#define PRES 1
#define TEMP 2

// Define the test field number
#define V 0
#define Q 1
#define W 2

typedef enum { STROUHAL, REYNOLDS, FROUDE, PECLET, HEATRELEASE, GAMMA, PTH, MU, K, CP, BETA, GRAVITY_DIRECTION, TOTAL_LOW_MACH_FLOW_PARAMETERS } LowMachFlowParametersTypes;
PETSC_EXTERN const char* lowMachFlowParametersTypeNames[TOTAL_LOW_MACH_FLOW_PARAMETERS + 1];

typedef struct {
    PetscReal strouhal;
    PetscReal reynolds;
    PetscReal froude;
    PetscReal peclet;
    PetscReal heatRelease;
    PetscReal gamma;
    PetscReal pth;  /* non-dimensional constant thermodynamic pressure */
    PetscReal mu;   /* non-dimensional viscosity */
    PetscReal k;    /* non-dimensional thermal conductivity */
    PetscReal cp;   /* non-dimensional specific heat capacity */
    PetscReal beta; /* non-dimensional thermal expansion coefficient */
    PetscInt gravityDirection;
} LowMachFlowParameters;

PETSC_EXTERN PetscErrorCode LowMachFlow_PackParameters(LowMachFlowParameters *parameters, PetscScalar *constantArray);
PETSC_EXTERN PetscErrorCode LowMachFlow_ParametersFromPETScOptions(PetscBag *flowParametersBag);

// Define the functions to setup the low mach flow
PETSC_EXTERN PetscErrorCode LowMachFlow_SetupDiscretization(DM dm);
PETSC_EXTERN PetscErrorCode LowMachFlow_StartProblemSetup(DM dm, PetscInt, PetscScalar []);
PETSC_EXTERN PetscErrorCode LowMachFlow_CompleteProblemSetup(TS ts, Vec *flowField);
PETSC_EXTERN PetscErrorCode LowMachFlow_CompleteFlowInitialization(DM dm, Vec u);

#endif
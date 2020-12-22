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

// Store the constant locations
#define NU 0
#define ALPHA 1

typedef struct {
    PetscBag parameters;
} LowMachFlowContext;

typedef struct {
    PetscReal nu;    /* Kinematic viscosity */
    PetscReal alpha; /* Thermal diffusivity */
    PetscReal T_in;  /* Inlet temperature*/
} Parameter;

// Setup
PetscErrorCode SetupDiscretization(DM dm, LowMachFlowContext *user);
PetscErrorCode SetupProblem(DM dm, LowMachFlowContext *user);
PetscErrorCode SetupParameters(LowMachFlowContext *user);
PetscErrorCode RemoveDiscretePressureNullspace(TS ts);

// Physics
void VIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);
void VIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]);
void WIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);
void WIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                    PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]);
void QIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);

#endif
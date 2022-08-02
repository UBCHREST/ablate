#if !defined(lowMachFlow_h)
#define lowMachFlow_h
#include <petsc.h>

// Define the test field number
typedef enum { VTEST, QTEST, WTEST } LowMachFlowTestFields;

typedef enum { STROUHAL, REYNOLDS, FROUDE, PECLET, HEATRELEASE, GAMMA, PTH, MU, K, CP, BETA, GRAVITY_DIRECTION, TOTAL_LOW_MACH_FLOW_PARAMETERS } LowMachFlowParametersTypes;
PETSC_EXTERN const char* lowMachFlowParametersTypeNames[TOTAL_LOW_MACH_FLOW_PARAMETERS + 1];

typedef enum { VEL, PRES, TEMP, TOTAL_LOW_MACH_FLOW_FIELDS } LowMachFlowFields;

typedef enum { MOM, MASS, ENERGY, TOTAL_LOW_MACH_SOURCE_FIELDS } LowMachSourceFields;

// Low Mach Kernels
PETSC_EXTERN void LowMachFlow_vIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                                     const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                     PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);

PETSC_EXTERN void LowMachFlow_vIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                                             const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[],
                                                             const PetscScalar a_x[], PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]);

PETSC_EXTERN void LowMachFlow_wIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                                     const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                     PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);

PETSC_EXTERN void LowMachFlow_wIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                                             const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[],
                                                             const PetscScalar a_x[], PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]);

PETSC_EXTERN void LowMachFlow_qIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                                     const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                     PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]);

PETSC_EXTERN void LowMachFlow_g0_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g1_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g0_qT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g0_qT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g1_qT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g0_vT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g0_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g1_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g3_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g2_vp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g0_wu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g0_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g1_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

PETSC_EXTERN void LowMachFlow_g3_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                                    const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]);

#endif
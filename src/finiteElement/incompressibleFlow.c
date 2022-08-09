#include "incompressibleFlow.h"
/*F
The incompressible flow formulation outlined in docs/content/formulations/incompressibleFlow
F*/

const char *incompressibleFlowParametersTypeNames[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS + 1] = {"strouhal", "reynolds", "peclet", "mu", "k", "cp", "unknown"};

// \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u}
void IncompressibleFlow_vIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                               const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    PetscInt Nc = dim;
    PetscInt c, d;

    // du/dt
    for (d = 0; d < dim; ++d) {
        f0[d] = constants[STROUHAL] * u_t[uOff[VEL] + d];  // rho is assumed unity
    }

    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            f0[c] += u[uOff[VEL] + d] * u_x[uOff_x[VEL] + c * dim + d];  // rho is assumed to be unity
        }
    }

    // Add in any fixed source term
    if (NfAux > 0) {
        for (d = 0; d < dim; ++d) {
            f0[d] += a[aOff[MOM] + d];
        }
    }
}

// \nabla^S \boldsymbol{v} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}(\boldsymbol{u}) - p \nabla \cdot \boldsymbol{v}
void IncompressibleFlow_vIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                                       const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                       PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]) {
    const PetscReal coefficient = constants[MU] / constants[REYNOLDS];
    const PetscInt Nc = dim;
    PetscInt c, d;

    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            f1[c * dim + d] = coefficient * (u_x[uOff_x[VEL] + c * dim + d] + u_x[uOff_x[VEL] + d * dim + c]);
        }
        f1[c * dim + c] -= u[uOff[PRES]];
    }
}

// \int_\Omega w \rho C_p S \frac{\partial T}{\partial t} + w \rho C_p \boldsymbol{u} \cdot \nabla T
void IncompressibleFlow_wIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                               const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    PetscInt d;

    f0[0] = constants[CP] * constants[STROUHAL] * u_t[uOff[TEMP]];

    for (d = 0; d < dim; ++d) {
        f0[0] += constants[CP] * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];  // rho is assumed unity
    }

    // Add in any fixed source term
    if (NfAux > 0) {
        f0[0] += a[aOff[ENERGY]];
    }
}

//  \nabla w \cdot \frac{k}{P} \nabla T
void IncompressibleFlow_wIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                                       const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                                       PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]) {
    const PetscReal coefficient = PetscRealPart(constants[K] / constants[PECLET]);
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        f1[d] = coefficient * u_x[uOff_x[TEMP] + d];
    }
}

/* \nabla\cdot u */
void IncompressibleFlow_qIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                               const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                               PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    PetscInt d;
    for (d = 0, f0[0] = 0.0; d < dim; ++d) {
        f0[0] += u_x[uOff_x[VEL] + d * dim + d];
    }

    // Add in any fixed source term
    if (NfAux > 0) {
        f0[0] += a[aOff[MASS]];
    }
}

/*Jacobians*/
// \int \phi_i \frac{\partial \psi_{u_c,j}}{\partial x_c}
void IncompressibleFlow_g1_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g1[d * dim + d] = 1.0;
    }
}

void IncompressibleFlow_g0_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    PetscInt c, d;
    const PetscInt Nc = dim;

    // \boldsymbol{\phi_i} \cdot \rho S \psi_j
    for (d = 0; d < dim; ++d) {
        g0[d * dim + d] = constants[STROUHAL] * u_tShift;
    }

    // \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \frac{\partial u_k}{\partial x_c}\hat{e}_k \right)
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g0[c * Nc + d] += u_x[uOff_x[VEL] + c * Nc + d];
        }
    }
}

// \boldsymbol{\phi_i} \cdot \rho u_c \nabla \psi_j
void IncompressibleFlow_g1_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    PetscInt NcI = dim;
    PetscInt NcJ = dim;
    PetscInt c, d, e;

    for (c = 0; c < NcI; ++c) {
        for (d = 0; d < NcJ; ++d) {
            for (e = 0; e < dim; ++e) {
                if (c == d) {
                    g1[(c * NcJ + d) * dim + e] += u[uOff[VEL] + e];
                }
            }
        }
    }
}

//  - \psi_{p,j} \nabla \cdot \boldsymbol{\phi_i}
void IncompressibleFlow_g2_vp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) g2[d * dim + d] = -1.0;
}

// \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l \right)
void IncompressibleFlow_g3_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]) {
    const PetscReal coefficient = PetscRealPart(constants[MU] / constants[REYNOLDS]);
    const PetscInt Nc = dim;
    PetscInt c, d;

    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g3[((c * Nc + c) * dim + d) * dim + d] += coefficient;  // gradU
            g3[((c * Nc + d) * dim + d) * dim + c] += coefficient;  // gradU transpose
        }
    }
}

// w \rho C_p S \frac{\partial T}{\partial t}
void IncompressibleFlow_g0_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP] * constants[STROUHAL]);
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g0[d] = coefficient * u_tShift;
    }
}

// \rho C_p \psi_{u_c,j} \frac{\partial T}{\partial x_c}
void IncompressibleFlow_g0_wu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP]);
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g0[d] = coefficient * u_x[uOff_x[TEMP] + d];
    }
}

//  \phi_i \rho C_p \boldsymbol{u} \cdot  \nabla \psi_{T,j}
void IncompressibleFlow_g1_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP]);
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g1[d] = coefficient * u[uOff[VEL] + d];
    }
}

//  \frac{k}{P} \nabla  \phi_i \cdot   \nabla \psi_{T,j}
void IncompressibleFlow_g3_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
                              const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]) {
    const PetscReal coefficient = PetscRealPart(constants[K] / constants[PECLET]);
    PetscInt d;

    for (d = 0; d < dim; ++d) {
        g3[d * dim + d] = coefficient;
    }
}
#include "lowMachFlow.h"
/*
The low Mach flow formulation outlined in docs/contents/formulations/lowMachFlow
*/

/* =q \left(-\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t} + \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u} - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right) */
static void qIntegrandTestFunction(PetscInt dim,CHKERRQ                           PetscInt Nf,CHKERRQ                           PetscInt NfAux,CHKERRQ                           const PetscInt uOff[],CHKERRQ                           const PetscInt uOff_x[],CHKERRQ                           const PetscScalar u[],CHKERRQ                           const PetscScalar u_t[],CHKERRQ                           const PetscScalar u_x[],CHKERRQ                           const PetscInt aOff[],CHKERRQ                           const PetscInt aOff_x[],CHKERRQ                           const PetscScalar a[],CHKERRQ                           const PetscScalar a_t[],CHKERRQ                           const PetscScalar a_x[],CHKERRQ                           PetscReal t,CHKERRQ                           const PetscReal X[],CHKERRQ                           PetscInt numConstants,CHKERRQ                           const PetscScalar constants[],CHKERRQ                           PetscScalar f0[]) {
    PetscInt d;

    // -\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t}
    f0[0] = -u_t[uOff[TEMP]] * constants[STROUHAL] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]);

    // \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u}
    for (d = 0; d < dim; ++d) {CHKERRQf0[0] += constants[PTH] / u[uOff[TEMP]] * u_x[uOff_x[VEL] + d * dim + d];
    }

    // - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right)
    for (d = 0; d < dim; ++d) {CHKERRQf0[0] -= constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
    }
}

/* \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \frac{\rho \hat{\boldsymbol{z}}}{F^2} \cdot
 * \boldsymbol{v} */
static void vIntegrandTestFunction(PetscInt dim,CHKERRQ                           PetscInt Nf,CHKERRQ                           PetscInt NfAux,CHKERRQ                           const PetscInt uOff[],CHKERRQ                           const PetscInt uOff_x[],CHKERRQ                           const PetscScalar u[],CHKERRQ                           const PetscScalar u_t[],CHKERRQ                           const PetscScalar u_x[],CHKERRQ                           const PetscInt aOff[],CHKERRQ                           const PetscInt aOff_x[],CHKERRQ                           const PetscScalar a[],CHKERRQ                           const PetscScalar a_t[],CHKERRQ                           const PetscScalar a_x[],CHKERRQ                           PetscReal t,CHKERRQ                           const PetscReal X[],CHKERRQ                           PetscInt numConstants,CHKERRQ                           const PetscScalar constants[],CHKERRQ                           PetscScalar f0[]) {
    PetscInt Nc = dim;
    PetscInt c, d;

    const PetscReal rho = constants[PTH] / u[uOff[TEMP]];

    // \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t}
    for (d = 0; d < dim; ++d) {CHKERRQf0[d] = rho * constants[STROUHAL] * u_t[uOff[VEL] + d];
    }

    // \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u}
    for (c = 0; c < Nc; ++c) {CHKERRQfor (d = 0; d < dim; ++d) {CHKERRQ    f0[c] += rho * u[uOff[VEL] + d] * u_x[uOff_x[VEL] + c * dim + d];CHKERRQ}
    }

    // rho \hat{z}/F^2
    f0[(PetscInt)constants[GRAVITY_DIRECTION]] += rho / (constants[FROUDE] * constants[FROUDE]);
}

/*.5 (\nabla \boldsymbol{v} + \nabla \boldsymbol{v}^T) \cdot 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I}) - p \nabla \cdot
 * \boldsymbol{v} */
static void vIntegrandTestGradientFunction(PetscInt dim,CHKERRQ                                   PetscInt Nf,CHKERRQ                                   PetscInt NfAux,CHKERRQ                                   const PetscInt uOff[],CHKERRQ                                   const PetscInt uOff_x[],CHKERRQ                                   const PetscScalar u[],CHKERRQ                                   const PetscScalar u_t[],CHKERRQ                                   const PetscScalar u_x[],CHKERRQ                                   const PetscInt aOff[],CHKERRQ                                   const PetscInt aOff_x[],CHKERRQ                                   const PetscScalar a[],CHKERRQ                                   const PetscScalar a_t[],CHKERRQ                                   const PetscScalar a_x[],CHKERRQ                                   PetscReal t,CHKERRQ                                   const PetscReal X[],CHKERRQ                                   PetscInt numConstants,CHKERRQ                                   const PetscScalar constants[],CHKERRQ                                   PetscScalar f1[]) {
    const PetscInt Nc = dim;
    PetscInt c, d;

    const PetscReal coefficient = constants[MU] / constants[REYNOLDS];  // (0.5 * 2.0)

    PetscReal u_divergence = 0.0;
    for (c = 0; c < Nc; ++c) {CHKERRQu_divergence += u_x[uOff_x[VEL] + c * dim + c];
    }

    // (\nabla \boldsymbol{v}) \cdot \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I})
    for (c = 0; c < Nc; ++c) {CHKERRQ// 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T)CHKERRQfor (d = 0; d < dim; ++d) {CHKERRQ    f1[c * dim + d] = 0.5 * coefficient * (u_x[uOff_x[VEL] + c * dim + d] + u_x[uOff_x[VEL] + d * dim + c]);CHKERRQ}
CHKERRQ// -1/3 (\nable \cdot \boldsybol{u}) \boldsymbol{I}CHKERRQf1[c * dim + c] -= coefficient / 3.0 * u_divergence;
    }

    // (\nabla \boldsymbol{v}^T) \cdot \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I})
    for (c = 0; c < Nc; ++c) {CHKERRQ// 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T)CHKERRQfor (d = 0; d < dim; ++d) {CHKERRQ    f1[d * dim + c] += 0.5 * coefficient * (u_x[uOff_x[VEL] + c * dim + d] + u_x[uOff_x[VEL] + d * dim + c]);CHKERRQ}
CHKERRQ// -1/3 (\nable \cdot \boldsybol{u}) \boldsymbol{I}CHKERRQf1[c * dim + c] -= coefficient / 3.0 * u_divergence;
    }

    // - p \nabla \cdot \boldsymbol{v}
    for (c = 0; c < Nc; ++c) {CHKERRQf1[c * dim + c] -= u[uOff[PRES]];
    }
}

/*w \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t} + w \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T */
static void wIntegrandTestFunction(PetscInt dim,CHKERRQ                           PetscInt Nf,CHKERRQ                           PetscInt NfAux,CHKERRQ                           const PetscInt uOff[],CHKERRQ                           const PetscInt uOff_x[],CHKERRQ                           const PetscScalar u[],CHKERRQ                           const PetscScalar u_t[],CHKERRQ                           const PetscScalar u_x[],CHKERRQ                           const PetscInt aOff[],CHKERRQ                           const PetscInt aOff_x[],CHKERRQ                           const PetscScalar a[],CHKERRQ                           const PetscScalar a_t[],CHKERRQ                           const PetscScalar a_x[],CHKERRQ                           PetscReal t,CHKERRQ                           const PetscReal X[],CHKERRQ                           PetscInt numConstants,CHKERRQ                           const PetscScalar constants[],CHKERRQ                           PetscScalar f0[]) {
    // \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t}
    f0[0] = constants[CP] * constants[STROUHAL] * constants[PTH] / u[uOff[TEMP]] * u_t[uOff[TEMP]];

    // \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T
    for (PetscInt d = 0; d < dim; ++d) {CHKERRQf0[0] += constants[CP] * constants[PTH] / u[uOff[TEMP]] * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
    }
}

/*  \nabla w \cdot \frac{k}{P} \nabla T */
static void wIntegrandTestGradientFunction(PetscInt dim,CHKERRQ                                   PetscInt Nf,CHKERRQ                                   PetscInt NfAux,CHKERRQ                                   const PetscInt uOff[],CHKERRQ                                   const PetscInt uOff_x[],CHKERRQ                                   const PetscScalar u[],CHKERRQ                                   const PetscScalar u_t[],CHKERRQ                                   const PetscScalar u_x[],CHKERRQ                                   const PetscInt aOff[],CHKERRQ                                   const PetscInt aOff_x[],CHKERRQ                                   const PetscScalar a[],CHKERRQ                                   const PetscScalar a_t[],CHKERRQ                                   const PetscScalar a_x[],CHKERRQ                                   PetscReal t,CHKERRQ                                   const PetscReal X[],CHKERRQ                                   PetscInt numConstants,CHKERRQ                                   const PetscScalar constants[],CHKERRQ                                   PetscScalar f1[]) {
    // \nabla w \cdot \frac{k}{P} \nabla T
    for (PetscInt d = 0; d < dim; ++d) {CHKERRQf1[d] = constants[K] / constants[PECLET] * u_x[uOff_x[TEMP] + d];
    }
}

/*Jacobians*/
static void g0_qu(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g0[]) {
    // - \phi_i \psi_{u_c,j} \frac{p^{th}}{T^2} \frac{\partial T}{\partial x_c}
    for (PetscInt d = 0; d < dim; ++d) {CHKERRQg0[d] = -constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_x[uOff_x[TEMP] + d];
    }
}

static void g1_qu(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g1[]) {
    PetscInt d;
    // \phi_i \frac{p^{th}}{T} \frac{\partial \psi_{u_c,j}}{\partial x_c}
    for (d = 0; d < dim; ++d) {CHKERRQg1[d * dim + d] = constants[PTH] / u[uOff[TEMP]];
    }
}

static void g0_qT(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g0[]) {
    // \frac{F_{q,i}}{\partial c_{\frac{\partial T}{\partial t},j}} =  \int \frac{-\phi_i S  p^{th}}{T^2} \psi_j
    g0[0] = -constants[STROUHAL] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_tShift;

    g0[0] += 2.0 * u_t[uOff[TEMP]] * constants[STROUHAL] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]] * u[uOff[TEMP]]);

    // \frac{\phi_i p^{th}}{T^2} \left( - \psi_{T,j}  \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \left(\frac{2}{T} \psi_{T,j} \nabla T\right) \right)
    for (PetscInt d = 0; d < dim; ++d) {CHKERRQg0[0] += constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * (-u_x[uOff_x[VEL] + d * dim + d] + 2.0 / u[uOff[TEMP]] * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d]);
    }
}

static void g1_qT(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g1[]) {
    // -\frac{\phi_i p^{th}}{T^2} \left( \boldsymbol{u}\cdot  \nabla \psi_{T,j}\right)
    for (PetscInt d = 0; d < dim; ++d) {CHKERRQg1[d] = -constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * (u[uOff[VEL] + d]);
    }
}

static void g0_vT(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g0[]) {
    PetscInt c, d;
    const PetscInt Nc = dim;

    // - \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \psi_{T,j}  S \frac{\partial \boldsymbol{u}}{\partial t}
    for (d = 0; d < dim; ++d) {CHKERRQg0[d] = -constants[PTH] * constants[STROUHAL] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_t[uOff[VEL] + d];
    }

    // - \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \psi_{T,j} \boldsymbol{u} \cdot \nabla \boldsymbol{u}
    for (c = 0; c < Nc; ++c) {CHKERRQfor (d = 0; d < dim; ++d) {CHKERRQ    g0[c] -= constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[VEL] + c * dim + d];CHKERRQ}
    }

    // -\frac{p^{th}}{T^2} \psi_{T,j} \frac{\hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{\phi_i}
    g0[(PetscInt)constants[GRAVITY_DIRECTION]] -= constants[PTH] / (constants[FROUDE] * constants[FROUDE] * u[uOff[TEMP]] * u[uOff[TEMP]]);
}

static void g0_vu(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g0[]) {
    PetscInt c, d;
    const PetscInt Nc = dim;

    // \frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{\frac{\partial u_c}{\partial t},j}} = \int_\Omega \boldsymbol{\phi_i} \cdot \rho S \psi_j
    for (d = 0; d < dim; ++d) {CHKERRQg0[d * dim + d] = constants[STROUHAL] * constants[PTH] / u[uOff[TEMP]] * u_tShift;
    }

    // \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \frac{\partial u_k}{\partial x_c}\hat{e}_k
    for (c = 0; c < Nc; ++c) {CHKERRQfor (d = 0; d < dim; ++d) {CHKERRQ    g0[c * Nc + d] += constants[PTH] / u[uOff[TEMP]] * u_x[uOff_x[VEL] + c * Nc + d];CHKERRQ}
    }
}

static void g1_vu(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g1[]) {
    PetscInt NcI = dim;
    PetscInt NcJ = dim;
    PetscInt c, d, e;

    // \phi_i \cdot \rho u_c \nabla \psi_j
    for (c = 0; c < NcI; ++c) {CHKERRQfor (d = 0; d < NcJ; ++d) {CHKERRQ    for (e = 0; e < dim; ++e) {CHKERRQ        if (c == d) {CHKERRQ            g1[(c * NcJ + d) * dim + e] += constants[PTH] / u[uOff[TEMP]] * u[uOff[VEL] + e];CHKERRQ        }CHKERRQ    }CHKERRQ}
    }
}

static void g3_vu(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g3[]) {
    const PetscInt Nc = dim;
    PetscInt c, d;

    // \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l
    // \right) - \frac{1}{3} \frac{\partial \psi_j}{\partial x_c}
    for (c = 0; c < Nc; ++c) {CHKERRQfor (d = 0; d < dim; ++d) {CHKERRQ    g3[((c * Nc + c) * dim + d) * dim + d] += constants[MU] / constants[REYNOLDS];  // gradUCHKERRQ    g3[((c * Nc + d) * dim + d) * dim + c] += constants[MU] / constants[REYNOLDS];  // gradU transpose
CHKERRQ    g3[((c * Nc + d) * dim + d) * dim + c] -= 2.0 / 3.0 * constants[MU] / constants[REYNOLDS];CHKERRQ}
    }
}

static void g2_vp(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g2[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) {CHKERRQg2[d * dim + d] = -1.0;
    }
}

static void g0_wu(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g0[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) {CHKERRQg0[d] = constants[CP] * constants[PTH] / u[uOff[TEMP]] * u_x[uOff_x[TEMP] + d];
    }
}

static void g0_wT(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g0[]) {
    //\frac{\partial F_{w,i}}{\partial c_{\frac{\partial T}{\partial t},j}} = \psi_i C_p S p^{th} \frac{1}{T} \psi_{j}
    g0[0] = constants[CP] * constants[STROUHAL] * constants[PTH] / u[uOff[TEMP]] * u_tShift;

    //- \phi_i C_p S p^{th} \frac{\partial T}{\partial t} \frac{1}{T^2} \psi_{T,j}  + ...
    g0[0] -= constants[CP] * constants[STROUHAL] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_t[uOff[TEMP]];

    // -\phi_i C_p p^{th} \boldsymbol{u} \cdot  \frac{\nabla T}{T^2} \psi_{T,j}
    for (PetscInt d = 0; d < dim; ++d) {CHKERRQg0[0] -= constants[CP] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
    }
}

static void g1_wT(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g1[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) {CHKERRQg1[d] = constants[CP] * constants[PTH] / u[uOff[TEMP]] * u[uOff[VEL] + d];
    }
}

static void g3_wT(PetscInt dim,CHKERRQ          PetscInt Nf,CHKERRQ          PetscInt NfAux,CHKERRQ          const PetscInt uOff[],CHKERRQ          const PetscInt uOff_x[],CHKERRQ          const PetscScalar u[],CHKERRQ          const PetscScalar u_t[],CHKERRQ          const PetscScalar u_x[],CHKERRQ          const PetscInt aOff[],CHKERRQ          const PetscInt aOff_x[],CHKERRQ          const PetscScalar a[],CHKERRQ          const PetscScalar a_t[],CHKERRQ          const PetscScalar a_x[],CHKERRQ          PetscReal t,CHKERRQ          PetscReal u_tShift,CHKERRQ          const PetscReal x[],CHKERRQ          PetscInt numConstants,CHKERRQ          const PetscScalar constants[],CHKERRQ          PetscScalar g3[]) {
    for (PetscInt d = 0; d < dim; ++d) {CHKERRQg3[d * dim + d] = constants[K] / constants[PECLET];
    }
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Nc; ++d) u[d] = 0.0;
    return 0;
}

static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Nc; ++d) {CHKERRQu[d] = 1.0;
    }
    return 0;
}

static PetscErrorCode createPressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace) {
    Vec vec;
    PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero, zero, zero};
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    if (ofield != PRES) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Nullspace must be for pressure field at correct index, not %D", ofield);
    funcs[nfield] = constant;
    ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
    ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec, "Pressure Null Space");CHKERRQ(ierr);
    ierr = VecViewFromOptions(vec, NULL, "-pressure_nullspace_view");CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, PRES, &vec, nullSpace);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode removeDiscretePressureNullspace(DM dm, Vec u) {
    MatNullSpace nullsp;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = createPressureNullSpace(dm, PRES, PRES, &nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullsp, u);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Make the discrete pressure discretely divergence free */
static PetscErrorCode removeDiscretePressureNullspaceOnTs(TS ts) {
    Vec u;
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
    ierr = removeDiscretePressureNullspace(dm, u);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode lowMachFlowSetupDiscretization(Flow flow) {
    DM cdm = flow->dm;
    PetscFE fe[3];
    MPI_Comm comm;
    PetscInt dim, cStart;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;

    // determine if it a simplex element and the number of dimensions
    DMPolytopeType ct;
    ierr = DMPlexGetHeightStratum(flow->dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(flow->dm, cStart, &ct);CHKERRQ(ierr);
    PetscBool simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;

    // Determine the number of dimensions
    ierr = DMGetDimension(flow->dm, &dim);CHKERRQ(ierr);

    /* Create finite element */
    ierr = PetscObjectGetComm((PetscObject)flow->dm, &comm);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(comm, dim, dim, simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe[VEL], "velocity");CHKERRQ(ierr);

    ierr = PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe[VEL], fe[PRES]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe[PRES], "pressure");CHKERRQ(ierr);

    ierr = PetscFECreateDefault(comm, dim, 1, simplex, "temp_", PETSC_DEFAULT, &fe[2]);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe[VEL], fe[TEMP]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe[TEMP], "temperature");CHKERRQ(ierr);

    /* Set discretization and boundary conditions for each mesh */
    ierr = DMSetField(flow->dm, VEL, NULL, (PetscObject)fe[VEL]);CHKERRQ(ierr);
    ierr = DMSetField(flow->dm, PRES, NULL, (PetscObject)fe[PRES]);CHKERRQ(ierr);
    ierr = DMSetField(flow->dm, TEMP, NULL, (PetscObject)fe[TEMP]);CHKERRQ(ierr);

    // Create the discrete systems for the DM based upon the fields added to the DM
    ierr = DMCreateDS(flow->dm);CHKERRQ(ierr);

    while (cdm) {CHKERRQierr = DMCopyDisc(flow->dm, cdm);CHKERRQCHKERRQ(ierr);CHKERRQierr = DMGetCoarseDM(cdm, &cdm);CHKERRQCHKERRQ(ierr);
    }

    // Clean up the fields
    ierr = PetscFEDestroy(&fe[VEL]);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe[PRES]);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe[TEMP]);CHKERRQ(ierr);

    {CHKERRQPetscObject pressure;CHKERRQMatNullSpace nullspacePres;
CHKERRQierr = DMGetField(flow->dm, PRES, NULL, &pressure);CHKERRQCHKERRQ(ierr);CHKERRQierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres);CHKERRQCHKERRQ(ierr);CHKERRQierr = PetscObjectCompose(pressure, "nullspace", (PetscObject)nullspacePres);CHKERRQCHKERRQ(ierr);CHKERRQierr = MatNullSpaceDestroy(&nullspacePres);CHKERRQCHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

static PetscErrorCode lowMachFlowStartProblemSetup(Flow flow) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscDS prob;
    ierr = DMGetDS(flow->dm, &prob);CHKERRQ(ierr);

    // V, W, Q Test Function
    ierr = PetscDSSetResidual(prob, V, vIntegrandTestFunction, vIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, W, wIntegrandTestFunction, wIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, Q, qIntegrandTestFunction, NULL);CHKERRQ(ierr);

    ierr = PetscDSSetJacobian(prob, V, VEL, g0_vu, g1_vu, NULL, g3_vu);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, V, PRES, NULL, NULL, g2_vp, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, V, TEMP, g0_vT, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, Q, VEL, g0_qu, g1_qu, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, Q, TEMP, g0_qT, g1_qT, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, W, VEL, g0_wu, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, W, TEMP, g0_wT, g1_wT, NULL, g3_wT);CHKERRQ(ierr);
    /* Setup constants */;
    {CHKERRQFlowParameters *param;CHKERRQPetscScalar constants[TOTAlCONSTANTS];
CHKERRQierr = PetscBagGetData(flow->parameters, (void **)&param);CHKERRQCHKERRQ(ierr);CHKERRQPackFlowParameters(param, constants);CHKERRQierr = PetscDSSetConstants(prob, TOTAlCONSTANTS, constants);CHKERRQCHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode lowMachFlowCompleteProblemSetup(Flow flow, TS ts) {
    PetscErrorCode ierr;
    FlowParameters *parameters;
    DM dm;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = PetscBagGetData(flow->parameters, (void **)&parameters);CHKERRQ(ierr);

    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm, &(flow->flowField));CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)(flow->flowField), "Numerical Solution");CHKERRQ(ierr);
    ierr = VecSetOptionsPrefix((flow->flowField), "num_sol_");CHKERRQ(ierr);

    ierr = DMSetNullSpaceConstructor(dm, PRES, createPressureNullSpace);CHKERRQ(ierr);

    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &parameters);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &parameters);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &parameters);CHKERRQ(ierr);

    ierr = TSSetPreStep(ts, removeDiscretePressureNullspaceOnTs);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode lowMachFlowDestroy(Flow flow) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr = VecDestroy(&(flow->flowField));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode LowMachFlowCreate(Flow flow) {
    PetscFunctionBeginUser;
    flow->setupDiscretization = lowMachFlowSetupDiscretization;
    flow->startProblemSetup = lowMachFlowStartProblemSetup;
    flow->completeProblemSetup = lowMachFlowCompleteProblemSetup;
    flow->completeFlowInitialization = removeDiscretePressureNullspace;
    flow->destroy = lowMachFlowDestroy;

    PetscFunctionReturn(0);
}

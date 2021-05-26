#include "lowMachFlow.h"

const char *lowMachFlowParametersTypeNames[TOTAL_LOW_MACH_FLOW_PARAMETERS + 1] = {
    "strouhal", "reynolds", "froude", "peclet", "heatRelease", "gamma", "pth", "mu", "k", "cp", "beta", "gravityDirection", "unknown"};

static const char *lowMachFlowFieldNames[TOTAL_LOW_MACH_FLOW_FIELDS + 1] = {"velocity", "pressure", "temperature", "unknown"};
static const char *lowMachSourceFieldNames[TOTAL_LOW_MACH_SOURCE_FIELDS + 1] = {"momentum_source", "mass_source", "energy_source", "unknown"};

/* =q \left(-\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t} + \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u} - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right) */
static void qIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[],
                                   PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    PetscInt d;

    // -\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t}
    f0[0] = -u_t[uOff[TEMP]] * constants[STROUHAL] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]);

    // \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u}
    for (d = 0; d < dim; ++d) {
        f0[0] += constants[PTH] / u[uOff[TEMP]] * u_x[uOff_x[VEL] + d * dim + d];
    }

    // - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right)
    for (d = 0; d < dim; ++d) {
        f0[0] -= constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
    }

    // Add in any fixed source term
    if (NfAux > 0) {
        f0[0] += a[aOff[MASS]];
    }
}

/* \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \frac{\rho \hat{\boldsymbol{z}}}{F^2} \cdot
 * \boldsymbol{v} */
static void vIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[],
                                   PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    PetscInt Nc = dim;
    PetscInt c, d;

    const PetscReal rho = constants[PTH] / u[uOff[TEMP]];

    // \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t}
    for (d = 0; d < dim; ++d) {
        f0[d] = rho * constants[STROUHAL] * u_t[uOff[VEL] + d];
    }

    // \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u}
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            f0[c] += rho * u[uOff[VEL] + d] * u_x[uOff_x[VEL] + c * dim + d];
        }
    }

    // rho \hat{z}/F^2
    f0[(PetscInt)constants[GRAVITY_DIRECTION]] += rho / (constants[FROUDE] * constants[FROUDE]);

    // Add in any fixed source term
    if (NfAux > 0) {
        for (d =0; d < dim; ++d){
            f0[d] += a[aOff[MOM] + d];
        }
    }
}

/*.5 (\nabla \boldsymbol{v} + \nabla \boldsymbol{v}^T) \cdot 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I}) - p \nabla \cdot
 * \boldsymbol{v} */
static void vIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                           const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]) {
    const PetscInt Nc = dim;
    PetscInt c, d;

    const PetscReal coefficient = constants[MU] / constants[REYNOLDS];  // (0.5 * 2.0)

    PetscReal u_divergence = 0.0;
    for (c = 0; c < Nc; ++c) {
        u_divergence += u_x[uOff_x[VEL] + c * dim + c];
    }

    // (\nabla \boldsymbol{v}) \cdot \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I})
    for (c = 0; c < Nc; ++c) {
        // 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T)
        for (d = 0; d < dim; ++d) {
            f1[c * dim + d] += 0.5 * coefficient * (u_x[uOff_x[VEL] + c * dim + d] + u_x[uOff_x[VEL] + d * dim + c]);
        }

        // -1/3 (\nable \cdot \boldsybol{u}) \boldsymbol{I}
        f1[c * dim + c] -= coefficient / 3.0 * u_divergence;
    }

    // (\nabla \boldsymbol{v}^T) \cdot \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I})
    for (c = 0; c < Nc; ++c) {
        // 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T)
        for (d = 0; d < dim; ++d) {
            f1[d * dim + c] += 0.5 * coefficient * (u_x[uOff_x[VEL] + c * dim + d] + u_x[uOff_x[VEL] + d * dim + c]);
        }

        // -1/3 (\nable \cdot \boldsybol{u}) \boldsymbol{I}
        f1[c * dim + c] -= coefficient / 3.0 * u_divergence;
    }

    // - p \nabla \cdot \boldsymbol{v}
    for (c = 0; c < Nc; ++c) {
        f1[c * dim + c] -= u[uOff[PRES]];
    }
}

/*w \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t} + w \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T */
static void wIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[],
                                   PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    // \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t}
    f0[0] = constants[CP] * constants[STROUHAL] * constants[PTH] / u[uOff[TEMP]] * u_t[uOff[TEMP]];

    // \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T
    for (PetscInt d = 0; d < dim; ++d) {
        f0[0] += constants[CP] * constants[PTH] / u[uOff[TEMP]] * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
    }

    // Add in any fixed source term
    if (NfAux > 0) {
        f0[0] += a[aOff[ENERGY]];
    }
}

/*  \nabla w \cdot \frac{k}{P} \nabla T */
static void wIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                           const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]) {
    // \nabla w \cdot \frac{k}{P} \nabla T
    for (PetscInt d = 0; d < dim; ++d) {
        f1[d] = constants[K] / constants[PECLET] * u_x[uOff_x[TEMP] + d];
    }
}

/*Jacobians*/
static void g0_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    // - \phi_i \psi_{u_c,j} \frac{p^{th}}{T^2} \frac{\partial T}{\partial x_c}
    for (PetscInt d = 0; d < dim; ++d) {
        g0[d] = -constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_x[uOff_x[TEMP] + d];
    }
}

static void g1_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    PetscInt d;
    // \phi_i \frac{p^{th}}{T} \frac{\partial \psi_{u_c,j}}{\partial x_c}
    for (d = 0; d < dim; ++d) {
        g1[d * dim + d] = constants[PTH] / u[uOff[TEMP]];
    }
}

static void g0_qT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    // \frac{F_{q,i}}{\partial c_{\frac{\partial T}{\partial t},j}} =  \int \frac{-\phi_i S  p^{th}}{T^2} \psi_j
    g0[0] = -constants[STROUHAL] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_tShift;

    g0[0] += 2.0 * u_t[uOff[TEMP]] * constants[STROUHAL] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]] * u[uOff[TEMP]]);

    // \frac{\phi_i p^{th}}{T^2} \left( - \psi_{T,j}  \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \left(\frac{2}{T} \psi_{T,j} \nabla T\right) \right)
    for (PetscInt d = 0; d < dim; ++d) {
        g0[0] += constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * (-u_x[uOff_x[VEL] + d * dim + d] + 2.0 / u[uOff[TEMP]] * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d]);
    }
}

static void g1_qT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    // -\frac{\phi_i p^{th}}{T^2} \left( \boldsymbol{u}\cdot  \nabla \psi_{T,j}\right)
    for (PetscInt d = 0; d < dim; ++d) {
        g1[d] = -constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * (u[uOff[VEL] + d]);
    }
}

static void g0_vT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    PetscInt c, d;
    const PetscInt Nc = dim;

    // - \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \psi_{T,j}  S \frac{\partial \boldsymbol{u}}{\partial t}
    for (d = 0; d < dim; ++d) {
        g0[d] = -constants[PTH] * constants[STROUHAL] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_t[uOff[VEL] + d];
    }

    // - \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \psi_{T,j} \boldsymbol{u} \cdot \nabla \boldsymbol{u}
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g0[c] -= constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[VEL] + c * dim + d];
        }
    }

    // -\frac{p^{th}}{T^2} \psi_{T,j} \frac{\hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{\phi_i}
    g0[(PetscInt)constants[GRAVITY_DIRECTION]] -= constants[PTH] / (constants[FROUDE] * constants[FROUDE] * u[uOff[TEMP]] * u[uOff[TEMP]]);
}

static void g0_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    PetscInt c, d;
    const PetscInt Nc = dim;

    // \frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{\frac{\partial u_c}{\partial t},j}} = \int_\Omega \boldsymbol{\phi_i} \cdot \rho S \psi_j
    for (d = 0; d < dim; ++d) {
        g0[d * dim + d] = constants[STROUHAL] * constants[PTH] / u[uOff[TEMP]] * u_tShift;
    }

    // \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \frac{\partial u_k}{\partial x_c}\hat{e}_k
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g0[c * Nc + d] += constants[PTH] / u[uOff[TEMP]] * u_x[uOff_x[VEL] + c * Nc + d];
        }
    }
}

static void g1_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    PetscInt NcI = dim;
    PetscInt NcJ = dim;
    PetscInt c, d, e;

    // \phi_i \cdot \rho u_c \nabla \psi_j
    for (c = 0; c < NcI; ++c) {
        for (d = 0; d < NcJ; ++d) {
            for (e = 0; e < dim; ++e) {
                if (c == d) {
                    g1[(c * NcJ + d) * dim + e] += constants[PTH] / u[uOff[TEMP]] * u[uOff[VEL] + e];
                }
            }
        }
    }
}

static void g3_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]) {
    const PetscInt Nc = dim;
    PetscInt c, d;

    // \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l
    // \right) - \frac{1}{3} \frac{\partial \psi_j}{\partial x_c}
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g3[((c * Nc + c) * dim + d) * dim + d] += constants[MU] / constants[REYNOLDS];  // gradU
            g3[((c * Nc + d) * dim + d) * dim + c] += constants[MU] / constants[REYNOLDS];  // gradU transpose

            g3[((c * Nc + d) * dim + c) * dim + d] -= 2.0 / 3.0 * constants[MU] / constants[REYNOLDS];
        }
    }
}

static void g2_vp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g2[d * dim + d] = -1.0;
    }
}

static void g0_wu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g0[d] = constants[CP] * constants[PTH] / u[uOff[TEMP]] * u_x[uOff_x[TEMP] + d];
    }
}

static void g0_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    //\frac{\partial F_{w,i}}{\partial c_{\frac{\partial T}{\partial t},j}} = \psi_i C_p S p^{th} \frac{1}{T} \psi_{j}
    g0[0] = constants[CP] * constants[STROUHAL] * constants[PTH] / u[uOff[TEMP]] * u_tShift;

    //- \phi_i C_p S p^{th} \frac{\partial T}{\partial t} \frac{1}{T^2} \psi_{T,j}  + ...
    g0[0] -= constants[CP] * constants[STROUHAL] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_t[uOff[TEMP]];

    // -\phi_i C_p p^{th} \boldsymbol{u} \cdot  \frac{\nabla T}{T^2} \psi_{T,j}
    for (PetscInt d = 0; d < dim; ++d) {
        g0[0] -= constants[CP] * constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
    }
}

static void g1_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g1[d] = constants[CP] * constants[PTH] / u[uOff[TEMP]] * u[uOff[VEL] + d];
    }
}

static void g3_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]) {
    for (PetscInt d = 0; d < dim; ++d) {
        g3[d * dim + d] = constants[K] / constants[PECLET];
    }
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Nc; ++d) u[d] = 0.0;
    return 0;
}

static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Nc; ++d) {
        u[d] = 1.0;
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

PetscErrorCode LowMachFlow_CompleteFlowInitialization(DM dm, Vec u) {
    MatNullSpace nullsp;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = createPressureNullSpace(dm, PRES, PRES, &nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullsp, u);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* Make the discrete pressure discretely divergence free */
static PetscErrorCode removeDiscretePressureNullspaceOnTs(TS ts, void* context) {
    Vec u;
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
    ierr = LowMachFlow_CompleteFlowInitialization(dm, u);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode LowMachFlow_SetupDiscretization(FlowData flowData, DM dm) {
    DM cdm = dm;
    PetscInt dim;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    //Store the field data
    flowData->dm = dm;
    ierr = DMSetApplicationContext(flowData->dm, flowData);CHKERRQ(ierr);

    // Determine the number of dimensions
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // Register each field, this order must match the order in LowMachFlowFields enum
    ierr = FlowRegisterField(flowData, lowMachFlowFieldNames[VEL], "vel_", dim, FE);CHKERRQ(ierr);
    ierr = FlowRegisterField(flowData, lowMachFlowFieldNames[PRES], "pres_", 1, FE);CHKERRQ(ierr);
    ierr = FlowRegisterField(flowData, lowMachFlowFieldNames[TEMP], "temp_", 1, FE);CHKERRQ(ierr);

    // Create the discrete systems for the DM based upon the fields added to the DM
    ierr = FlowFinalizeRegisterFields(flowData);CHKERRQ(ierr);

    while (cdm) {
        ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
        ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
    }

    {
        PetscObject pressure;
        MatNullSpace nullspacePres;

        ierr = DMGetField(dm, PRES, NULL, &pressure);CHKERRQ(ierr);
        ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres);CHKERRQ(ierr);
        ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject)nullspacePres);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspacePres);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode LowMachFlow_EnableAuxFields(FlowData flowData) {
    PetscFunctionBeginUser;
    // Determine the number of dimensions
    PetscInt dim;
    PetscErrorCode ierr = DMGetDimension(flowData->dm, &dim);CHKERRQ(ierr);

    ierr = FlowRegisterAuxField(flowData, lowMachSourceFieldNames[MOM] , "momentum_source_", dim, FE);CHKERRQ(ierr);
    ierr = FlowRegisterAuxField(flowData, lowMachSourceFieldNames[MASS] , "mass_source_", 1, FE);CHKERRQ(ierr);
    ierr = FlowRegisterAuxField(flowData, lowMachSourceFieldNames[ENERGY] , "energy_source_", 1, FE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode LowMachFlow_StartProblemSetup(FlowData flowData, PetscInt numberParameters, PetscScalar parameters[]) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);CHKERRQ(ierr);

    // V, W, Q Test Function
    ierr = PetscDSSetResidual(prob, VTEST, vIntegrandTestFunction, vIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, WTEST, wIntegrandTestFunction, wIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, QTEST, qIntegrandTestFunction, NULL);CHKERRQ(ierr);

    ierr = PetscDSSetJacobian(prob, VTEST, VEL, g0_vu, g1_vu, NULL, g3_vu);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, VTEST, PRES, NULL, NULL, g2_vp, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, VTEST, TEMP, g0_vT, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, QTEST, VEL, g0_qu, g1_qu, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, QTEST, TEMP, g0_qT, g1_qT, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, WTEST, VEL, g0_wu, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, WTEST, TEMP, g0_wT, g1_wT, NULL, g3_wT);CHKERRQ(ierr);

    /* Setup constants */;
    {
        if (numberParameters != TOTAL_LOW_MACH_FLOW_PARAMETERS) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "wrong number of flow parameters");
        }
        ierr = PetscDSSetConstants(prob, TOTAL_LOW_MACH_FLOW_PARAMETERS, parameters);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode LowMachFlow_CompleteProblemSetup(FlowData flowData, TS ts) {
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBeginUser;
    ierr =  FlowCompleteProblemSetup(flowData, ts);CHKERRQ(ierr);

    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = DMSetNullSpaceConstructor(dm, PRES, createPressureNullSpace);CHKERRQ(ierr);
    ierr = FlowRegisterPreStep(flowData, removeDiscretePressureNullspaceOnTs, NULL);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode LowMachFlow_PackParameters(LowMachFlowParameters *parameters, PetscScalar *constantArray) {
    constantArray[STROUHAL] = parameters->strouhal;
    constantArray[REYNOLDS] = parameters->reynolds;
    constantArray[FROUDE] = parameters->froude;
    constantArray[PECLET] = parameters->peclet;
    constantArray[HEATRELEASE] = parameters->heatRelease;
    constantArray[PTH] = parameters->pth;
    constantArray[GAMMA] = parameters->gamma;
    constantArray[MU] = parameters->mu;
    constantArray[K] = parameters->k;
    constantArray[CP] = parameters->cp;
    constantArray[BETA] = parameters->beta;
    constantArray[GRAVITY_DIRECTION] = parameters->gravityDirection;
    return 0;
}

PetscErrorCode LowMachFlow_ParametersFromPETScOptions(PetscBag *flowParametersBag) {
    LowMachFlowParameters *p;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    // create an empty bag
    ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(LowMachFlowParameters), flowParametersBag);CHKERRQ(ierr);

    // setup PETSc parameter bag
    ierr = PetscBagGetData(*flowParametersBag, (void **)&p);CHKERRQ(ierr);
    ierr = PetscBagSetName(*flowParametersBag, "flowParameters", "Low Mach Flow Parameters");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->strouhal, 1.0, lowMachFlowParametersTypeNames[STROUHAL], "Strouhal number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->reynolds, 1.0, lowMachFlowParametersTypeNames[REYNOLDS], "Reynolds number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->froude, 1.0, lowMachFlowParametersTypeNames[FROUDE], "Froude number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->peclet, 1.0, lowMachFlowParametersTypeNames[PECLET], "Peclet number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->heatRelease, 1.0, lowMachFlowParametersTypeNames[HEATRELEASE], "Heat Release number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->gamma, 1.0, lowMachFlowParametersTypeNames[GAMMA], "gamma: p_o/(\rho_o Cp_o T_o");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->pth, 1.0, lowMachFlowParametersTypeNames[PTH], "non-dimensional thermodyamic pressure");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->mu, 1.0, lowMachFlowParametersTypeNames[MU], "non-dimensional viscosity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->k, 1.0, lowMachFlowParametersTypeNames[K], "non-dimensional thermal conductivity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->cp, 1.0, lowMachFlowParametersTypeNames[CP], "non-dimensional specific heat capacity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->beta, 1.0, lowMachFlowParametersTypeNames[BETA], "non-dimensional thermal expansion coefficient ");CHKERRQ(ierr);
    ierr = PetscBagRegisterInt(*flowParametersBag, &p->gravityDirection, 0, lowMachFlowParametersTypeNames[GRAVITY_DIRECTION], "axis");CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
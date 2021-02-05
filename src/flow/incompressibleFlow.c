#include "incompressibleFlow.h"
/*F
The incompressible flow formulation outlined in docs/conents/formulations/incompressibleFlow
F*/


// \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u}
static void vIntegrandTestFunction(PetscInt dim,
                            PetscInt Nf,
                            PetscInt NfAux,
                            const PetscInt uOff[],
                            const PetscInt uOff_x[],
                            const PetscScalar u[],
                            const PetscScalar u_t[],
                            const PetscScalar u_x[],
                            const PetscInt aOff[],
                            const PetscInt aOff_x[],
                            const PetscScalar a[],
                            const PetscScalar a_t[],
                            const PetscScalar a_x[],
                            PetscReal t,
                            const PetscReal X[],
                            PetscInt numConstants,
                            const PetscScalar constants[],
                            PetscScalar f0[]) {
    PetscInt Nc = dim;
    PetscInt c, d;

    // du/dt
    for (d = 0; d < dim; ++d) {
        f0[d] = constants[STROUHAL]*u_t[uOff[VEL] + d];// rho is assumed unity
    }

    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            f0[c] += u[uOff[VEL] + d] * u_x[uOff_x[VEL]+ c * dim + d];// rho is assumed to be unity
        }
    }
}

// \nabla^S \boldsymbol{v} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}(\boldsymbol{u}) - p \nabla \cdot \boldsymbol{v}
static void vIntegrandTestGradientFunction(PetscInt dim,
                                    PetscInt Nf,
                                    PetscInt NfAux,
                                    const PetscInt uOff[],
                                    const PetscInt uOff_x[],
                                    const PetscScalar u[],
                                    const PetscScalar u_t[],
                                    const PetscScalar u_x[],
                                    const PetscInt aOff[],
                                    const PetscInt aOff_x[],
                                    const PetscScalar a[],
                                    const PetscScalar a_t[],
                                    const PetscScalar a_x[],
                                    PetscReal t,
                                    const PetscReal X[],
                                    PetscInt numConstants,
                                    const PetscScalar constants[],
                                    PetscScalar f1[]) {
    const PetscReal coefficient = constants[MU]/constants[REYNOLDS];
    const PetscInt Nc = dim;
    PetscInt c, d;

    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            f1[c * dim + d] = coefficient * (u_x[uOff_x[VEL] +c * dim + d] + u_x[uOff_x[VEL] + d * dim + c]);
        }
        f1[c * dim + c] -= u[uOff[PRES]];
    }
}

// \int_\Omega w \rho C_p S \frac{\partial T}{\partial t} + w \rho C_p \boldsymbol{u} \cdot \nabla T
static void wIntegrandTestFunction(PetscInt dim,
                            PetscInt Nf,
                            PetscInt NfAux,
                            const PetscInt uOff[],
                            const PetscInt uOff_x[],
                            const PetscScalar u[],
                            const PetscScalar u_t[],
                            const PetscScalar u_x[],
                            const PetscInt aOff[],
                            const PetscInt aOff_x[],
                            const PetscScalar a[],
                            const PetscScalar a_t[],
                            const PetscScalar a_x[],
                            PetscReal t,
                            const PetscReal X[],
                            PetscInt numConstants,
                            const PetscScalar constants[],
                            PetscScalar f0[]) {
    PetscInt d;

    f0[0] = constants[CP]*constants[STROUHAL]*u_t[uOff[TEMP]];
    for (d = 0; d < dim; ++d) {
        f0[0] +=  constants[STROUHAL]* u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];// rho is assumed unity
    }
}

//  \nabla w \cdot \frac{k}{P} \nabla T
static void wIntegrandTestGradientFunction(PetscInt dim,
                                    PetscInt Nf,
                                    PetscInt NfAux,
                                    const PetscInt uOff[],
                                    const PetscInt uOff_x[],
                                    const PetscScalar u[],
                                    const PetscScalar u_t[],
                                    const PetscScalar u_x[],
                                    const PetscInt aOff[],
                                    const PetscInt aOff_x[],
                                    const PetscScalar a[],
                                    const PetscScalar a_t[],
                                    const PetscScalar a_x[],
                                    PetscReal t,
                                    const PetscReal X[],
                                    PetscInt numConstants,
                                    const PetscScalar constants[],
                                    PetscScalar f1[]) {
    const PetscReal coefficient = PetscRealPart(constants[K]/constants[PECLET] );
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        f1[d] = coefficient * u_x[uOff_x[TEMP] + d];
    }
}

/* \nabla\cdot u */
static void qIntegrandTestFunction(PetscInt dim,
                            PetscInt Nf,
                            PetscInt NfAux,
                            const PetscInt uOff[],
                            const PetscInt uOff_x[],
                            const PetscScalar u[],
                            const PetscScalar u_t[],
                            const PetscScalar u_x[],
                            const PetscInt aOff[],
                            const PetscInt aOff_x[],
                            const PetscScalar a[],
                            const PetscScalar a_t[],
                            const PetscScalar a_x[],
                            PetscReal t,
                            const PetscReal X[],
                            PetscInt numConstants,
                            const PetscScalar constants[],
                            PetscScalar f0[]) {
    PetscInt d;
    for (d = 0, f0[0] = 0.0; d < dim; ++d) {
        f0[0] += u_x[uOff_x[VEL] + d * dim + d];
    }
}

/*Jacobians*/
// \int \phi_i \frac{\partial \psi_{u_c,j}}{\partial x_c}
static void g1_qu(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g1[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d){
        g1[d * dim + d] = 1.0;
    }
}

static void g0_vu(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g0[]) {
    PetscInt c, d;
    const PetscInt Nc = dim;

    // \boldsymbol{\phi_i} \cdot \rho S \psi_j
    for (d = 0; d < dim; ++d){
        g0[d * dim + d] = constants[STROUHAL]*u_tShift;
    }

    // \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \frac{\partial u_k}{\partial x_c}\hat{e}_k \right)
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g0[c * Nc + d] += u_x[uOff_x[VEL] + c * Nc + d];
        }
    }
}

// \boldsymbol{\phi_i} \cdot \rho u_c \nabla \psi_j
static void g1_vu(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g1[]) {
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
static void g2_vp(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g2[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) g2[d * dim + d] = -1.0;
}

// \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l \right)
static void g3_vu(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g3[]) {
    const PetscReal coefficient = PetscRealPart(constants[MU]/constants[REYNOLDS]);
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
static void g0_wT(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g0[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP]*constants[STROUHAL]);
    PetscInt d;
    for (d = 0; d < dim; ++d){
        g0[d] = coefficient * u_tShift;
    }
}

// \rho C_p \psi_{u_c,j} \frac{\partial T}{\partial x_c}
static void g0_wu(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g0[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP]);
    PetscInt d;
    for (d = 0; d < dim; ++d){
        g0[d] = coefficient * u_x[uOff_x[TEMP] + d];
    }
}

//  \phi_i \rho C_p \boldsymbol{u} \cdot  \nabla \psi_{T,j}
static void g1_wT(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g1[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP]);
    PetscInt d;
    for (d = 0; d < dim; ++d){
        g1[d] = coefficient * u[uOff[VEL] + d];

    }
}

//  \frac{k}{P} \nabla  \phi_i \cdot   \nabla \psi_{T,j}
static void g3_wT(PetscInt dim,
                  PetscInt Nf,
                  PetscInt NfAux,
                  const PetscInt uOff[],
                  const PetscInt uOff_x[],
                  const PetscScalar u[],
                  const PetscScalar u_t[],
                  const PetscScalar u_x[],
                  const PetscInt aOff[],
                  const PetscInt aOff_x[],
                  const PetscScalar a[],
                  const PetscScalar a_t[],
                  const PetscScalar a_x[],
                  PetscReal t,
                  PetscReal u_tShift,
                  const PetscReal x[],
                  PetscInt numConstants,
                  const PetscScalar constants[],
                  PetscScalar g3[]) {
    const PetscReal coefficient = PetscRealPart(constants[K]/constants[PECLET] );
    PetscInt d;

    for (d = 0; d < dim; ++d){
        g3[d * dim + d] = coefficient;
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

PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace) {
    Vec vec;
    PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero, zero, zero};
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    if (ofield != 1) SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Nullspace must be for pressure field at index 1, not %D", ofield);
    funcs[nfield] = constant;
    ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
    ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vec, "Pressure Null Space");CHKERRQ(ierr);
    ierr = VecViewFromOptions(vec, NULL, "-pressure_nullspace_view");CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode removeDiscretePressureNullspace(DM dm, Vec u) {
    MatNullSpace nullsp;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = CreatePressureNullSpace(dm, PRES, PRES, &nullsp);CHKERRQ(ierr);
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

static PetscErrorCode incompressibleFlowSetupDiscretization(Flow flow) {
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

    while (cdm) {
        ierr = DMCopyDisc(flow->dm, cdm);CHKERRQ(ierr);
        ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
    }

    // Clean up the fields
    ierr = PetscFEDestroy(&fe[VEL]);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe[PRES]);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe[TEMP]);CHKERRQ(ierr);

    {
        PetscObject pressure;
        MatNullSpace nullspacePres;

        ierr = DMGetField(flow->dm, PRES, NULL, &pressure);CHKERRQ(ierr);
        ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres);CHKERRQ(ierr);
        ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject)nullspacePres);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspacePres);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

static PetscErrorCode incompressibleFlowStartProblemSetup(Flow flow) {
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
    ierr = PetscDSSetJacobian(prob, Q, VEL, NULL, g1_qu, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, W, VEL, g0_wu, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, W, TEMP, g0_wT, g1_wT, NULL, g3_wT);CHKERRQ(ierr);
    /* Setup constants */;
    {
        FlowParameters *param;
        PetscScalar constants[TOTAlCONSTANTS];

        ierr = PetscBagGetData(flow->parameters, (void **)&param);CHKERRQ(ierr);
        PackFlowParameters(param, constants);
        ierr = PetscDSSetConstants(prob, TOTAlCONSTANTS, constants);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

static PetscErrorCode incompressibleFlowCompleteProblemSetup(Flow flow, TS ts) {
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

    ierr = DMSetNullSpaceConstructor(dm, PRES, CreatePressureNullSpace);CHKERRQ(ierr);

    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &parameters);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &parameters);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &parameters);CHKERRQ(ierr);

    ierr = TSSetPreStep(ts, removeDiscretePressureNullspaceOnTs);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode incompressibleFlowDestroy(Flow flow) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr = VecDestroy(&(flow->flowField));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode IncompressibleFlowCreate(Flow flow) {

    PetscFunctionBeginUser;
    flow->setupDiscretization = incompressibleFlowSetupDiscretization;
    flow->startProblemSetup = incompressibleFlowStartProblemSetup;
    flow->completeProblemSetup = incompressibleFlowCompleteProblemSetup;
    flow->completeFlowInitialization = removeDiscretePressureNullspace;
    flow->destroy = incompressibleFlowDestroy;

    PetscFunctionReturn(0);
}

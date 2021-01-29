#include "lowMachFlow.h"
#include "parameters.h"
/*F
This Low Mach flow is time-dependent iso viscous Navier-Stokes flow. We discretize using the
finite element method on an unstructured mesh. The weak form equations are in the documentation
F*/

PetscErrorCode SetupDiscretization(DM dm, LowMachFlowContext *user) {
    DM cdm = dm;
    PetscFE fe[3];
    MPI_Comm comm;
    PetscInt dim, cStart;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;

    // determine if it a simplex element and the number of dimensions
    DMPolytopeType ct;
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
    PetscBool simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;

    // Determine the number of dimensions
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    /* Create finite element */
    ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(comm, dim, dim, simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe[VEL], "velocity");CHKERRQ(ierr);

    ierr = PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe[VEL], fe[PRES]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe[PRES], "pressure");CHKERRQ(ierr);

    ierr = PetscFECreateDefault(comm, dim, 1, simplex, "temp_", PETSC_DEFAULT, &fe[2]);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe[VEL], fe[TEMP]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fe[TEMP], "temperature");CHKERRQ(ierr);

    /* Set discretization and boundary conditions for each mesh */
    ierr = DMSetField(dm, VEL, NULL, (PetscObject)fe[VEL]);CHKERRQ(ierr);
    ierr = DMSetField(dm, PRES, NULL, (PetscObject)fe[PRES]);CHKERRQ(ierr);
    ierr = DMSetField(dm, TEMP, NULL, (PetscObject)fe[TEMP]);CHKERRQ(ierr);

    // Create the discrete systems for the DM based upon the fields added to the DM
    ierr = DMCreateDS(dm);CHKERRQ(ierr);

    while (cdm) {
        ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
        ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
    }

    // Clean up the fields
    ierr = PetscFEDestroy(&fe[VEL]);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe[PRES]);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe[TEMP]);CHKERRQ(ierr);

//    {
//        PetscObject pressure;
//        MatNullSpace nullspacePres;
//
//        ierr = DMGetField(dm, PRES, NULL, &pressure);CHKERRQ(ierr);
//        ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres);CHKERRQ(ierr);
//        ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject)nullspacePres);CHKERRQ(ierr);
//        ierr = MatNullSpaceDestroy(&nullspacePres);CHKERRQ(ierr);
//    }

    PetscFunctionReturn(0);
}

/* =q \left(-\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t} + \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u} - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right) */
void QIntegrandTestFunction(PetscInt dim,
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

    // -\frac{Sp^{th}}{T^2}\frac{\partial T}{\partial t}
    f0[0] = -u_t[uOff[TEMP]]*constants[STROUHAL]*constants[PTH]/(u[uOff[TEMP]]*u[uOff[TEMP]]);

    // \frac{p^{th}}{T} \nabla \cdot \boldsymbol{u}
    for (d = 0; d < dim; ++d) {
        f0[0] += constants[PTH]/u[uOff[TEMP]] *  u_x[uOff_x[VEL] + d * dim + d];
    }

    // - \frac{p^{th}}{T^2}\boldsymbol{u} \cdot \nabla T \right)
    for (d = 0; d < dim; ++d) {
        f0[0] -= constants[PTH]/(u[uOff[TEMP]]*u[uOff[TEMP]]) *  u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];
    }
}

/* \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u} + \frac{\rho \hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{v} */
void VIntegrandTestFunction(PetscInt dim,
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

    const PetscReal rho = constants[PTH]/u[uOff[TEMP]];

    // \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t}
    for (d = 0; d < dim; ++d) {
        f0[d] = rho*constants[STROUHAL] * u_t[uOff[VEL] + d];
    }

    // \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u}
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            f0[c] += rho*u[uOff[VEL] + d] * u_x[uOff_x[VEL]+ c * dim + d];
        }
    }

    // rho \hat{z}/F^2
    f0[(PetscInt)constants[GRAVITY_DIRECTION]] += rho/(constants[FROUDE]*constants[FROUDE]);
}

/*.5 (\nabla \boldsymbol{v} + \nabla \boldsymbol{v}^T) \cdot 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I}) - p \nabla \cdot \boldsymbol{v} */
void VIntegrandTestGradientFunction(PetscInt dim,
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

    const PetscInt Nc = dim;
    PetscInt c, d;

    const PetscReal coefficient = constants[MU]/constants[REYNOLDS];// (0.5 * 2.0)

    PetscReal u_divergence = 0.0;
    for (c = 0; c < Nc; ++c) {
        u_divergence += u_x[uOff_x[VEL] + c * dim + c];
    }

    // (\nabla \boldsymbol{v}) \cdot \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I})
    for (c = 0; c < Nc; ++c) {
        // 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T)
        for (d = 0; d < dim; ++d) {
            f1[c * dim + d] =  0.5 * coefficient * (u_x[uOff_x[VEL] + c * dim + d] + u_x[uOff_x[VEL] + d * dim + c]);
        }

        // -1/3 (\nable \cdot \boldsybol{u}) \boldsymbol{I}
        f1[c * dim + c] -= coefficient/3.0 * u_divergence;
    }

    // (\nabla \boldsymbol{v}^T) \cdot \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T) - 1/3 (\nabla \cdot \bolsymbol{u})\boldsymbol{I})
    for (c = 0; c < Nc; ++c) {
        // 2 \mu/R (.5 (\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T)
        for (d = 0; d < dim; ++d) {
            f1[d * dim + c] += 0.5 * coefficient * (u_x[uOff_x[VEL] + c * dim + d] + u_x[uOff_x[VEL] + d * dim + c]);
        }

        // -1/3 (\nable \cdot \boldsybol{u}) \boldsymbol{I}
        f1[c * dim + c] -= coefficient/3.0 * u_divergence;
    }

    // - p \nabla \cdot \boldsymbol{v}
    for (c = 0; c < Nc; ++c) {
        f1[c * dim + c] -= u[uOff[PRES]];
    }
}

/*w \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t} + w \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T */
void WIntegrandTestFunction(PetscInt dim,
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
    // \frac{C_p S p^{th}}{T} \frac{\partial T}{\partial t}
    f0[0] = constants[CP]*constants[STROUHAL]*constants[PTH]/u[uOff[TEMP]]*u_t[uOff[TEMP]];

    // \frac{C_p p^{th}}{T} \boldsymbol{u} \cdot \nabla T
    for (PetscInt d = 0; d < dim; ++d) {
        f0[0] += constants[CP]*constants[PTH]/u[uOff[TEMP]] *  u[uOff[VEL] + d]*u_x[uOff_x[TEMP] + d];
    }
}

/*  \nabla w \cdot \frac{k}{P} \nabla T */
void WIntegrandTestGradientFunction(PetscInt dim,
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
    // \nabla w \cdot \frac{k}{P} \nabla T
    for (PetscInt d = 0; d < dim; ++d) {
        f1[d] = constants[K]/constants[PECLET] * u_x[uOff_x[TEMP] + d];
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

PetscErrorCode RemoveDiscretePressureNullspace(DM dm, Vec u) {
    MatNullSpace nullsp;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = CreatePressureNullSpace(dm, PRES, PRES, &nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullsp, u);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

///* Make the discrete pressure discretely divergence free */
PetscErrorCode RemoveDiscretePressureNullspaceOnTs(TS ts) {
    Vec u;
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
    ierr = RemoveDiscretePressureNullspace(dm, u);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*Jacobians*/
static void g0_qu(PetscInt dim,
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
    // - \phi_i \psi_{u_c,j} \frac{p^{th}}{T^2} \frac{\partial T}{\partial x_c}
    for (PetscInt d = 0; d < dim; ++d) {
        g0[d] = -constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * u_x[uOff_x[TEMP] + d];
    }
}

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
    // \phi_i \frac{p^{th}}{T} \frac{\partial \psi_{u_c,j}}{\partial x_c}
    for (d = 0; d < dim; ++d){
        g1[d * dim + d] = constants[PTH]/u[uOff[TEMP]];
    }
}

static void g0_qT(PetscInt dim,
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

    // \frac{F_{q,i}}{\partial c_{\frac{\partial T}{\partial t},j}} =  \int \frac{-\phi_i S  p^{th}}{T^2} \psi_j
    g0[0] = - constants[STROUHAL] * constants[PTH]/(u[uOff[TEMP]] * u[uOff[TEMP]]) * u_tShift;

    g0[0] += 2.0*u_t[uOff[TEMP]]*constants[STROUHAL]*constants[PTH]/(u[uOff[TEMP]] * u[uOff[TEMP]] * u[uOff[TEMP]]);

    // \frac{\phi_i p^{th}}{T^2} \left( - \psi_{T,j}  \nabla \cdot \boldsymbol{u} + \boldsymbol{u}\cdot \left(\frac{2}{T} \psi_{T,j} \nabla T\right) \right)
    for (PetscInt d = 0; d < dim; ++d) {
        g0[0] += constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) *(-u_x[uOff_x[VEL] + d * dim + d] + 2.0/u[uOff[TEMP]] * u[uOff[VEL] + d ] * u_x[uOff_x[TEMP] + d]);
    }
}

static void g1_qT(PetscInt dim,
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
    // -\frac{\phi_i p^{th}}{T^2} \left( \boldsymbol{u}\cdot  \nabla \psi_{T,j}\right)
    for (PetscInt d = 0; d < dim; ++d) {
        g1[d] = -constants[PTH] / (u[uOff[TEMP]] * u[uOff[TEMP]]) * (u[uOff[VEL] + d ] );
    }
}

static void g0_vT(PetscInt dim,
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

    // - \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \psi_{T,j}  S \frac{\partial \boldsymbol{u}}{\partial t}
    for (d = 0; d < dim; ++d){
        g0[d] = -constants[PTH]*constants[STROUHAL]/(u[uOff[TEMP]]*u[uOff[TEMP]])*u_t[uOff[VEL]+d];
    }

    // - \boldsymbol{\phi_i} \cdot \frac{p^{th}}{T^2} \psi_{T,j} \boldsymbol{u} \cdot \nabla \boldsymbol{u}
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g0[c] -= constants[PTH]/(u[uOff[TEMP]]*u[uOff[TEMP]])*u[uOff[VEL] + d] * u_x[uOff_x[VEL]+ c * dim + d];
        }
    }

    // -\frac{p^{th}}{T^2} \psi_{T,j} \frac{\hat{\boldsymbol{z}}}{F^2} \cdot \boldsymbol{\phi_i}
    g0[(PetscInt)constants[GRAVITY_DIRECTION]] -= constants[PTH]/(constants[FROUDE]*constants[FROUDE]*u[uOff[TEMP]]*u[uOff[TEMP]]);
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

    // \frac{\partial F_{\boldsymbol{v}_i}}{\partial c_{\frac{\partial u_c}{\partial t},j}} = \int_\Omega \boldsymbol{\phi_i} \cdot \rho S \psi_j
    for (d = 0; d < dim; ++d){
        g0[d * dim + d] = constants[STROUHAL]*constants[PTH]/u[uOff[TEMP]]* u_tShift;
    }

    // \boldsymbol{\phi_i} \cdot \left(\rho \psi_j \frac{\partial u_k}{\partial x_c}\hat{e}_k
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g0[c * Nc + d] += constants[PTH]/u[uOff[TEMP]]*u_x[uOff_x[VEL] + c * Nc + d];
        }
    }
}

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

    // \phi_i \cdot \rho u_c \nabla \psi_j
    for (c = 0; c < NcI; ++c) {
        for (d = 0; d < NcJ; ++d) {
            for (e = 0; e < dim; ++e) {
                if (c == d) {
                    g1[(c * NcJ + d) * dim + e] += constants[PTH]/u[uOff[TEMP]]*u[uOff[VEL] + e];
                }
            }
        }
    }
}

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

    const PetscInt Nc = dim;
    PetscInt c, d;

    // \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l \right) - \frac{1}{3} \frac{\partial \psi_j}{\partial x_c}
    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            g3[((c * Nc + c) * dim + d) * dim + d] += constants[MU]/constants[REYNOLDS];  // gradU
            g3[((c * Nc + d) * dim + d) * dim + c] += constants[MU]/constants[REYNOLDS];  // gradU transpose

            g3[((c * Nc + d) * dim + d) * dim + c] -= 2.0/3.0*constants[MU]/constants[REYNOLDS];
        }
    }
}

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
    for (d = 0; d < dim; ++d){
        g2[d * dim + d] = -1.0;
    }
}

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
    PetscInt d;
    for (d = 0; d < dim; ++d){
        g0[d] = constants[CP]*constants[PTH]/u[uOff[TEMP]]*u_x[uOff_x[TEMP] + d];
    }
}


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
    //\frac{\partial F_{w,i}}{\partial c_{\frac{\partial T}{\partial t},j}} = \psi_i C_p S p^{th} \frac{1}{T} \psi_{j}
    g0[0] = constants[CP]*constants[STROUHAL]*constants[PTH]/u[uOff[TEMP]]*u_tShift;

    //- \phi_i C_p S p^{th} \frac{\partial T}{\partial t} \frac{1}{T^2} \psi_{T,j}  + ...
    g0[0] -= constants[CP]*constants[STROUHAL]*constants[PTH]/(u[uOff[TEMP]]*u[uOff[TEMP]])*u_t[uOff[TEMP]];

    // -\phi_i C_p p^{th} \boldsymbol{u} \cdot  \frac{\nabla T}{T^2} \psi_{T,j}
    for (PetscInt d = 0; d < dim; ++d){
        g0[0] -= constants[CP]*constants[PTH]/(u[uOff[TEMP]]*u[uOff[TEMP]])*u[uOff[VEL]+d]*u_x[uOff_x[TEMP]+d];
    }
}

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
    PetscInt d;
    for (d = 0; d < dim; ++d){
        g1[d] = constants[CP]*constants[PTH]/u[uOff[TEMP]]*u[uOff[VEL] + d];
    }
}

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
    for (PetscInt d = 0; d < dim; ++d){
        g3[d * dim + d] = constants[K]/constants[PECLET];
    }
}

PetscErrorCode StartProblemSetup(DM dm, LowMachFlowContext *ctx) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscDS prob;
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);

    // V, W, Q Test Function
    ierr = PetscDSSetResidual(prob, V, VIntegrandTestFunction, VIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, W, WIntegrandTestFunction, WIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, Q, QIntegrandTestFunction, NULL);CHKERRQ(ierr);

    ierr = PetscDSSetJacobian(prob, V, VEL, g0_vu, g1_vu, NULL, g3_vu);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, V, PRES, NULL, NULL, g2_vp, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, V, TEMP, g0_vT, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, Q, VEL, g0_qu, g1_qu, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, Q, TEMP, g0_qT, g1_qT, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, W, VEL, g0_wu, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, W, TEMP, g0_wT, g1_wT, NULL, g3_wT);CHKERRQ(ierr);

    /* Setup constants */;
    {
        FlowParameters *param;
        PetscScalar constants[TOTAlCONSTANTS];

        ierr = PetscBagGetData(ctx->parameters, (void **)&param);CHKERRQ(ierr);
        PackFlowParameters(param, constants);
        ierr = PetscDSSetConstants(prob, TOTAlCONSTANTS, constants);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode CompleteProblemSetup(TS ts, Vec *u, LowMachFlowContext *context) {
    PetscErrorCode ierr;
    FlowParameters *parameters;
    DM dm;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = PetscBagGetData(context->parameters, (void **)&parameters);CHKERRQ(ierr);

    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm, u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)*u, "Numerical Solution");CHKERRQ(ierr);
    ierr = VecSetOptionsPrefix(*u, "num_sol_");CHKERRQ(ierr);

//    ierr = DMSetNullSpaceConstructor(dm, PRES, CreatePressureNullSpace);CHKERRQ(ierr);

    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &parameters);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &parameters);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &parameters);CHKERRQ(ierr);

//    ierr = TSSetPreStep(ts, RemoveDiscretePressureNullspaceOnTs);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
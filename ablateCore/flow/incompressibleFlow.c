#include "incompressibleFlow.h"
/*F
The incompressible flow formulation outlined in docs/content/formulations/incompressibleFlow
F*/

const char *incompressibleFlowParametersTypeNames[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS + 1] = {"strouhal", "reynolds", "peclet", "mu", "k", "cp", "unknown"};
static const char *incompressibleFlowFieldNames[TOTAL_INCOMPRESSIBLE_FLOW_FIELDS + 1] = {"velocity", "pressure", "temperature", "unknown"};

// \boldsymbol{v} \cdot \rho S \frac{\partial \boldsymbol{u}}{\partial t} + \boldsymbol{v} \cdot \rho \boldsymbol{u} \cdot \nabla \boldsymbol{u}
static void vIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[],
                                   PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
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
}

// \nabla^S \boldsymbol{v} \cdot \frac{2 \mu}{R} \boldsymbol{\epsilon}(\boldsymbol{u}) - p \nabla \cdot \boldsymbol{v}
static void vIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
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
static void wIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[],
                                   PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    PetscInt d;

    f0[0] = constants[CP] * constants[STROUHAL] * u_t[uOff[TEMP]];

    for (d = 0; d < dim; ++d) {
        f0[0] += constants[CP] * u[uOff[VEL] + d] * u_x[uOff_x[TEMP] + d];  // rho is assumed unity
    }
}

//  \nabla w \cdot \frac{k}{P} \nabla T
static void wIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[],
                                           const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[]) {
    const PetscReal coefficient = PetscRealPart(constants[K] / constants[PECLET]);
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        f1[d] = coefficient * u_x[uOff_x[TEMP] + d];
    }
}

/* \nabla\cdot u */
static void qIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal X[],
                                   PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    PetscInt d;
    for (d = 0, f0[0] = 0.0; d < dim; ++d) {
        f0[0] += u_x[uOff_x[VEL] + d * dim + d];
    }
}

/*Jacobians*/
// \int \phi_i \frac{\partial \psi_{u_c,j}}{\partial x_c}
static void g1_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g1[d * dim + d] = 1.0;
    }
}

static void g0_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
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
static void g1_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
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
static void g2_vp(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[]) {
    PetscInt d;
    for (d = 0; d < dim; ++d) g2[d * dim + d] = -1.0;
}

// \nabla^S \boldsymbol{\phi_i} \cdot \frac{2 \mu}{R} \left( \frac{1}{2}\left( \hat{e}_l \frac{\partial \psi_j}{\partial x_l}\hat{e}_c + \hat{e}_c \frac{\partial \psi_j}{\partial x_l}\hat{e}_l \right)
static void g3_vu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]) {
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
static void g0_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP] * constants[STROUHAL]);
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g0[d] = coefficient * u_tShift;
    }
}

// \rho C_p \psi_{u_c,j} \frac{\partial T}{\partial x_c}
static void g0_wu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP]);
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g0[d] = coefficient * u_x[uOff_x[TEMP] + d];
    }
}

//  \phi_i \rho C_p \boldsymbol{u} \cdot  \nabla \psi_{T,j}
static void g1_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[]) {
    const PetscReal coefficient = PetscRealPart(constants[CP]);
    PetscInt d;
    for (d = 0; d < dim; ++d) {
        g1[d] = coefficient * u[uOff[VEL] + d];
    }
}

//  \frac{k}{P} \nabla  \phi_i \cdot   \nabla \psi_{T,j}
static void g3_wT(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[],
                  PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]) {
    const PetscReal coefficient = PetscRealPart(constants[K] / constants[PECLET]);
    PetscInt d;

    for (d = 0; d < dim; ++d) {
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

static PetscErrorCode createPressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace) {
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

PetscErrorCode IncompressibleFlow_CompleteFlowInitialization(DM dm, Vec u) {
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
    ierr = IncompressibleFlow_CompleteFlowInitialization(dm, u);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode IncompressibleFlow_SetupDiscretization(FlowData flowData, DM dm) {
    DM cdm = dm;
    PetscFE fe[3];
    MPI_Comm comm;
    PetscInt dim, cStart;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;

    //Store the field data
    flowData->dm = dm;

    // Determine the number of dimensions
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // Register each field, this order must match the order in IncompressibleFlowFields enum
    ierr = FlowRegisterFields(flowData, incompressibleFlowFieldNames[VEL], "vel_",  dim);CHKERRQ(ierr);
    ierr = FlowRegisterFields(flowData, incompressibleFlowFieldNames[PRES], "pres_",  1);CHKERRQ(ierr);
    ierr = FlowRegisterFields(flowData, incompressibleFlowFieldNames[TEMP], "temp_",  1);CHKERRQ(ierr);

    // Create the discrete systems for the DM based upon the fields added to the DM
    ierr = DMCreateDS(dm);CHKERRQ(ierr);

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

PetscErrorCode IncompressibleFlow_StartProblemSetup(FlowData flowData, PetscInt numberParameters, PetscScalar parameters[]) {
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
    ierr = PetscDSSetJacobian(prob, QTEST, VEL, NULL, g1_qu, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, WTEST, VEL, g0_wu, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, WTEST, TEMP, g0_wT, g1_wT, NULL, g3_wT);CHKERRQ(ierr);
    /* Setup constants */;
    {
        if (numberParameters != TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "wrong number of flow parameters");
        }
        ierr = PetscDSSetConstants(prob, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, parameters);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode IncompressibleFlow_CompleteProblemSetup(FlowData flowData, TS ts) {
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm, &(flowData->flowField));CHKERRQ(ierr);

    ierr = DMSetNullSpaceConstructor(dm, PRES, createPressureNullSpace);CHKERRQ(ierr);

    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL);CHKERRQ(ierr);

    ierr = TSSetPreStep(ts, removeDiscretePressureNullspaceOnTs);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode IncompressibleFlow_PackParameters(IncompressibleFlowParameters *parameters, PetscScalar *constantArray) {
    constantArray[STROUHAL] = parameters->strouhal;
    constantArray[REYNOLDS] = parameters->reynolds;
    constantArray[PECLET] = parameters->peclet;
    constantArray[MU] = parameters->mu;
    constantArray[K] = parameters->k;
    constantArray[CP] = parameters->cp;
    return 0;
}

PetscErrorCode IncompressibleFlow_ParametersFromPETScOptions(PetscBag *flowParametersBag) {
    IncompressibleFlowParameters *p;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    // create an empty bag
    ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(IncompressibleFlowParameters), flowParametersBag);CHKERRQ(ierr);

    // setup PETSc parameter bag
    ierr = PetscBagGetData(*flowParametersBag, (void **)&p);CHKERRQ(ierr);
    ierr = PetscBagSetName(*flowParametersBag, "flowParameters", "Low Mach Flow Parameters");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->strouhal, 1.0, incompressibleFlowParametersTypeNames[STROUHAL], "Strouhal number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->reynolds, 1.0, incompressibleFlowParametersTypeNames[REYNOLDS], "Reynolds number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->peclet, 1.0, incompressibleFlowParametersTypeNames[PECLET], "Peclet number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->mu, 1.0, incompressibleFlowParametersTypeNames[MU], "non-dimensional viscosity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->k, 1.0, incompressibleFlowParametersTypeNames[K], "non-dimensional thermal conductivity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->cp, 1.0, incompressibleFlowParametersTypeNames[CP], "non-dimensional specific heat capacity");CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
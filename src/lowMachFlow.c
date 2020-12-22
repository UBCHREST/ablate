#include "lowMachFlow.h"

/*F
This Low Mach flow is time-dependent isoviscous Navier-Stokes flow. We discretize using the
finite element method on an unstructured mesh. The weak form equations are

\begin{align*}
    < q, \nabla\cdot u > = 0
    <v, du/dt> + <v, u \cdot \nabla u> + < \nabla v, \nu (\nabla u + {\nabla u}^T) > - < \nabla\cdot v, p >  - < v, f  >  = 0
    < w, dT/dt > + < w, u \cdot \nabla T > + < \nabla w, \alpha \nabla T > - < w, Q > = 0
\end{align*}

where $\nu$ is the kinematic viscosity and $\alpha$ is thermal diffusivity.

For visualization, use

  -dm_view hdf5:$PWD/sol.h5 -sol_vec_view hdf5:$PWD/sol.h5::append -exact_vec_view hdf5:$PWD/sol.h5::append
F*/

PetscErrorCode SetupDiscretization(DM dm, LowMachFlowContext *user)
{
    DM              cdm   = dm;
    PetscFE         fe[3];
    MPI_Comm        comm;
    PetscInt        dim, cStart;
    PetscErrorCode  ierr;

    PetscFunctionBeginUser;

    // determine if it a simplex element and the number of dimensions
    DMPolytopeType ct;
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
    PetscBool simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;

    // Determine the number of dimensions
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    /* Create finite element */
    ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(comm, dim, dim, simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe[VEL], "velocity");CHKERRQ(ierr);

    ierr = PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe[VEL], fe[PRES]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe[PRES], "pressure");CHKERRQ(ierr);

    ierr = PetscFECreateDefault(comm, dim, 1, simplex, "temp_", PETSC_DEFAULT, &fe[2]);CHKERRQ(ierr);
    ierr = PetscFECopyQuadrature(fe[VEL], fe[TEMP]);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe[TEMP], "temperature");CHKERRQ(ierr);

    /* Set discretization and boundary conditions for each mesh */
    ierr = DMSetField(dm, VEL, NULL, (PetscObject) fe[VEL]);CHKERRQ(ierr);
    ierr = DMSetField(dm, PRES, NULL, (PetscObject) fe[PRES]);CHKERRQ(ierr);
    ierr = DMSetField(dm, TEMP, NULL, (PetscObject) fe[TEMP]);CHKERRQ(ierr);

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

    {
        PetscObject  pressure;
        MatNullSpace nullspacePres;

        ierr = DMGetField(dm, PRES, NULL, &pressure);CHKERRQ(ierr);
        ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullspacePres);CHKERRQ(ierr);
        ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullspacePres);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&nullspacePres);CHKERRQ(ierr);
    }

    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode SetupParameters(LowMachFlowContext *user)
{
    PetscBag       bag;
    Parameter     *p;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    /* setup PETSc parameter bag */
    ierr = PetscBagGetData(user->parameters, (void **) &p);CHKERRQ(ierr);
    ierr = PetscBagSetName(user->parameters, "par", "Low Mach flow parameters");CHKERRQ(ierr);
    bag  = user->parameters;
    ierr = PetscBagRegisterReal(bag, &p->nu,    1.0, "nu",    "Kinematic viscosity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(bag, &p->alpha, 1.0, "alpha", "Thermal diffusivity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(bag, &p->T_in,  1.0, "T_in",  "Inlet temperature");CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* f0_v = du/dt + u \cdot \nabla u */
void VIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) {
    const PetscReal nu = PetscRealPart(constants[NU]);
    PetscInt Nc = dim;
    PetscInt c, d;

    // du/dt
    for (d = 0; d < dim; ++d){
        f0[d] = u_t[uOff[VEL] + d];
    }

    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d){
            //TODO: add uOff and uOff_x
            f0[c] += u[d]*u_x[c*dim+d];
        }
    }
}

/*f1_v = \nu[grad(u) + grad(u)^T] - pI */
void VIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
    const PetscReal nu = PetscRealPart(constants[NU]);
    const PetscInt    Nc = dim;
    PetscInt        c, d;

    for (c = 0; c < Nc; ++c) {
        for (d = 0; d < dim; ++d) {
            //TODO: add uOff and uOff_x
            f1[c*dim+d] = nu*(u_x[c*dim+d] + u_x[d*dim+c]);
        }
        f1[c*dim+c] -= u[uOff[PRES]];
    }
}

/* f0_w = dT/dt + u.grad(T)*/
void WIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
    PetscInt d;

    f0[0] = u_t[uOff[TEMP]];
    for (d = 0; d < dim; ++d){
        f0[0] += u[uOff[VEL]+d]*u_x[uOff_x[TEMP]+d];
    }
}

void WIntegrandTestGradientFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
    const PetscReal alpha = PetscRealPart(constants[ALPHA]);
    PetscInt d;
    for (d = 0; d < dim; ++d){
        f1[d] = alpha*u_x[uOff_x[TEMP]+d];
    }
}

/* \nabla\cdot u */
void QIntegrandTestFunction(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
    PetscInt d;
    for (d = 0, f0[0] = 0.0; d < dim; ++d){
        f0[0] += u_x[d*dim+d];
    }
}

static PetscErrorCode constant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
    PetscInt d;
    for (d = 0; d < Nc; ++d){
        u[d] = 1.0;
    }
    return 0;
}

static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt ofield, PetscInt nfield, MatNullSpace *nullSpace)
{
    Vec              vec;
    PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {zero, zero, zero};
    PetscErrorCode   ierr;

    PetscFunctionBeginUser;
    if (ofield != 1) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Nullspace must be for pressure field at index 1, not %D", ofield);
    funcs[nfield] = constant;
    ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
    ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) vec, "Pressure Null Space");CHKERRQ(ierr);
    ierr = VecViewFromOptions(vec, NULL, "-pressure_nullspace_view");CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject) dm), PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode SetupProblem(DM dm, LowMachFlowContext *ctx) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscDS prob;
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);

    // V, W, Q Test Function
    ierr = PetscDSSetResidual(prob, V, VIntegrandTestFunction, VIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, W, WIntegrandTestFunction, WIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, Q, QIntegrandTestFunction, NULL);CHKERRQ(ierr);

    /* Setup constants */
    {
        Parameter *param;
        PetscScalar constants[2];

        ierr = PetscBagGetData(ctx->parameters, (void **) &param);CHKERRQ(ierr);
        constants[NU] = param->nu;
        constants[ALPHA] = param->alpha;
        ierr = PetscDSSetConstants(prob, 2, constants);CHKERRQ(ierr);
    }

    ierr = DMSetNullSpaceConstructor(dm, 1, CreatePressureNullSpace);CHKERRQ(ierr);

    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

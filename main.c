static char help[] = "Time-dependent Low Mach Flow in 2d channels with finite elements.\n\
We solve the Low Mach flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include "lowMachFlow.h"
#include "mesh.h"

static PetscErrorCode quadratic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
    u[0] = time + X[0]*X[0] + X[1]*X[1];
    u[1] = time + 2.0*X[0]*X[0] - 2.0*X[0]*X[1];
    return 0;
}
static PetscErrorCode quadratic_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode quadratic_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode quadratic_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx)
{
    T[0] = 1.0;
    return 0;
}

/* f0_v = du/dt - f */
static void f0_quadratic_v(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
    VIntegrandTestFunction(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    const PetscReal nu = PetscRealPart(constants[NU]);

    f0[0] -= (t*(2*X[0] + 2*X[1]) + 2*X[0]*X[0]*X[0] + 4*X[0]*X[0]*X[1] - 2*X[0]*X[1]*X[1] - 4.0*nu + 2);
    f0[1] -= (t*(2*X[0] - 2*X[1]) + 4*X[0]*X[1]*X[1] + 2*X[0]*X[0]*X[1] - 2*X[1]*X[1]*X[1] - 4.0*nu + 2);
}

/* f0_w = dT/dt + u.grad(T) - Q */
static void f0_quadratic_w(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
    WIntegrandTestFunction(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] += - (2*t + 1 + 3*X[0]*X[0] - 2*X[0]*X[1] + X[1]*X[1]);
}

static PetscErrorCode SetInitialConditions(TS ts, Vec u)
{
    DM             dm;
    PetscReal      t;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, t, u, NULL);CHKERRQ(ierr);
    ierr = RemoveDiscretePressureNullspace_Private(ts, u);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void            *ctxs[3];
    DM               dm;
    PetscDS          ds;
    Vec              v;
    PetscReal        ferrors[3];
    PetscInt         f;
    PetscErrorCode   ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

    for (f = 0; f < 3; ++f) {ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);CHKERRQ(ierr);}
    ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int) step, (double) crtime, (double) ferrors[0], (double) ferrors[1], (double) ferrors[2]);CHKERRQ(ierr);

    ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
    //ierr = TSGetSolution(ts, &u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);

    ierr = DMGetGlobalVector(dm, &v);CHKERRQ(ierr);
    // ierr = VecSet(v, 0.0);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, v);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) v, "Exact Solution");CHKERRQ(ierr);
    ierr = VecViewFromOptions(v, NULL, "-exact_vec_view");CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &v);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

extern int ex76(int argc,char **args);
int main(int argc,char **args){
    ex76(argc, args);
}

int main2(int argc,char **args)
{
    DM                  dm;   /* problem definition */
    TS                  ts;   /* timestepper */
    Vec                 u;    /* solution */
    LowMachFlowContext  context; /* user-defined work context */
    PetscReal           t;
    PetscErrorCode      ierr;

    // initialize petsc and mpi
    ierr = PetscInitialize(&argc, &args, NULL,help);if (ierr) return ierr;

    // setup and initialize the constant field variables
    ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameters), &context.parameters);CHKERRQ(ierr);
    ierr = SetupParameters(&context);CHKERRQ(ierr);

    // setup the ts
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
    ierr = CreateMesh(PETSC_COMM_WORLD, &dm, PETSC_TRUE, 2);CHKERRQ(ierr);
    ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
    ierr = DMSetApplicationContext(dm, &context);CHKERRQ(ierr);

    // setup problem
    ierr = SetupDiscretization(dm, &context);CHKERRQ(ierr);
    ierr = SetupProblem(dm, &context);CHKERRQ(ierr);

    // Override problem with source terms, boundary, and set the exact solution
    {
        PetscDS prob;
        ierr = DMGetDS(dm, &prob);
        CHKERRQ(ierr);

        // V, W Test Function
        ierr = PetscDSSetResidual(prob, V, f0_quadratic_v, VIntegrandTestGradientFunction);CHKERRQ(ierr);
        ierr = PetscDSSetResidual(prob, W, f0_quadratic_w, WIntegrandTestGradientFunction);CHKERRQ(ierr);

        Parameters *parameters;
        ierr = PetscBagGetData(context.parameters, (void **) &parameters);CHKERRQ(ierr);

        /* Setup Boundary Conditions */
        PetscInt         id;
        id   = 3;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity",    "marker", VEL, 0, NULL, (void (*)(void)) quadratic_u, (void (*)(void)) quadratic_u_t, 1, &id, parameters);CHKERRQ(ierr);
        id   = 1;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void)) quadratic_u, (void (*)(void)) quadratic_u_t, 1, &id, parameters);CHKERRQ(ierr);
        id   = 2;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall velocity",  "marker", VEL, 0, NULL, (void (*)(void)) quadratic_u, (void (*)(void)) quadratic_u_t, 1, &id, parameters);CHKERRQ(ierr);
        id   = 4;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity",   "marker", VEL, 0, NULL, (void (*)(void)) quadratic_u, (void (*)(void)) quadratic_u_t, 1, &id, parameters);CHKERRQ(ierr);
        id   = 3;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp",    "marker", TEMP, 0, NULL, (void (*)(void)) quadratic_T, (void (*)(void)) quadratic_T_t, 1, &id, parameters);CHKERRQ(ierr);
        id   = 1;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void)) quadratic_T, (void (*)(void)) quadratic_T_t, 1, &id, parameters);CHKERRQ(ierr);
        id   = 2;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp",  "marker", TEMP, 0, NULL, (void (*)(void)) quadratic_T, (void (*)(void)) quadratic_T_t, 1, &id, parameters);CHKERRQ(ierr);
        id   = 4;
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp",   "marker", TEMP, 0, NULL, (void (*)(void)) quadratic_T, (void (*)(void)) quadratic_T_t, 1, &id, parameters);CHKERRQ(ierr);


        // Set the exact solution
        ierr = PetscDSSetExactSolution(prob, VEL, quadratic_u, parameters);CHKERRQ(ierr);
        ierr = PetscDSSetExactSolution(prob, PRES, quadratic_p, parameters);CHKERRQ(ierr);
        ierr = PetscDSSetExactSolution(prob, TEMP, quadratic_T, parameters);CHKERRQ(ierr);
        ierr = PetscDSSetExactSolutionTimeDerivative(prob, VEL, quadratic_u_t, parameters);CHKERRQ(ierr);
        ierr = PetscDSSetExactSolutionTimeDerivative(prob, PRES, NULL, parameters);CHKERRQ(ierr);
        ierr = PetscDSSetExactSolutionTimeDerivative(prob, TEMP, quadratic_T_t, parameters);CHKERRQ(ierr);


        ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
        ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);

        ierr = DMSetNullSpaceConstructor(dm, PRES, CreatePressureNullSpace);CHKERRQ(ierr);

        ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &parameters);CHKERRQ(ierr);
        ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &parameters);CHKERRQ(ierr);
        ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &parameters);CHKERRQ(ierr);
    }

    // Setup the TS
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetPreStep(ts, RemoveDiscretePressureNullspace);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    // Set initial conditions from the exact solution
    ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);CHKERRQ(ierr); /* Must come after SetFromOptions() */
    ierr = SetInitialConditions(ts, u);CHKERRQ(ierr);

    ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
    ierr = DMSetOutputSequenceNumber(dm, 0, t);CHKERRQ(ierr);
    ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts, MonitorError, &context, NULL);CHKERRQ(ierr);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);
    ierr = TSSolve(ts, u);CHKERRQ(ierr);

    // Compare the actual vs expected values
    ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);

    // Cleanup
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
    ierr = PetscBagDestroy(&context.parameters);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
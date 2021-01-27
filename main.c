static char help[] =
    "Time-dependent Low Mach Flow in 2d channels with finite elements.\n\
We solve the Low Mach flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include "lowMachFlow.h"
#include "mesh.h"
#include "parameters.h"


static PetscErrorCode SetInitialConditions(TS ts, Vec u) {
    DM dm;
    PetscReal t;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = TSGetTime(ts, &t);
    CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, t, u, NULL);
    CHKERRQ(ierr);
//    ierr = RemoveDiscretePressureNullspace(dm, u);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void *ctxs[3];
    DM dm;
    PetscDS ds;
    Vec v;
    PetscReal ferrors[3];
    PetscInt f;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    for (f = 0; f < 3; ++f) {
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
    ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int)step, (double)crtime, (double)ferrors[0], (double)ferrors[1], (double)ferrors[2]);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = DMGetGlobalVector(dm, &u);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = TSGetSolution(ts, &u);CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscObjectSetName((PetscObject)u, "Numerical Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMRestoreGlobalVector(dm, &u);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = DMGetGlobalVector(dm, &v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = VecSet(v, 0.0);CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscObjectSetName((PetscObject)v, "Exact Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecViewFromOptions(v, NULL, "-exact_vec_view");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMRestoreGlobalVector(dm, &v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    PetscFunctionReturn(0);
}

/*
  CASE: quadratic
  In 2D we use exact solution:


*/
static PetscErrorCode quadratic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    // u = {t + x^2 + y^2, t + 2*x^2 + 2*x*y}
    u[0] = time + X[0] * X[0] + X[1] * X[1];
    u[1] = time + 2.0 * X[0] * X[0] + 2.0 * X[0] * X[1];
    return 0;
}
static PetscErrorCode quadratic_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode quadratic_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    // p = x + y - 1
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode quadratic_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    // T = t + x + y
    T[0] = time + X[0] + X[1] + 1;
    return 0;
}
static PetscErrorCode quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

static PetscReal Power(PetscReal x, PetscInt exp){
    return PetscPowReal(x, exp);
}

static void f0_quadratic_q(PetscInt dim,
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
                           PetscScalar f0[]){
    QIntegrandTestFunction(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    const PetscReal S = constants[STROUHAL];
    const PetscReal Pth = constants[PTH];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= -((Pth*S)/Power(1 + t + x + y,2)) + (4*Pth*x)/(1 + t + x + y) -
             (Pth*(t + 2*Power(x,2) + 2*x*y))/Power(1 + t + x + y,2) -
             (Pth*(t + Power(x,2) + Power(y,2)))/Power(1 + t + x + y,2);
}

static void f0_quadratic_v(PetscInt dim,
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
    VIntegrandTestFunction(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal S = constants[STROUHAL];
    const PetscReal Pth = constants[PTH];
    const PetscReal mu = constants[MU];
    const PetscReal R = constants[REYNOLDS];
    const PetscReal F = constants[FROUDE];

    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= 1 - (5.333333333333334*mu)/R + (Pth*S)/(1 + t + x + y) + (2*Pth*y*(t + 2*Power(x,2) + 2*x*y))/(1 + t + x + y) + (2*Pth*x*(t + Power(x,2) + Power(y,2)))/(1 + t + x + y);
    f0[1] -= 1 - (4.*mu)/R + Pth/(Power(F,2)*(1 + t + x + y)) + (Pth*S)/(1 + t + x + y) + (2*Pth*x*(t + 2*Power(x,2) + 2*x*y))/(1 + t + x + y) + (Pth*(4*x + 2*y)*(t + Power(x,2) + Power(y,2)))/(1 + t + x + y);
}

/* f0_w = dT/dt + u.grad(T) - Q */
static void f0_quadratic_w(PetscInt dim,
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
    WIntegrandTestFunction(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal S = constants[STROUHAL];
    const PetscReal Pth = constants[PTH];
    const PetscReal Cp = constants[CP];
    const PetscReal H = constants[HEATRELEASE];

    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= (Cp*Pth*(S + 2*t + 3*Power(x,2) + 2*x*y + Power(y,2)))/(H*S*(1 + t + x + y));
}

int main(int argc, char **args) {
  DM dm;                      /* problem definition */
  TS ts;                      /* timestepper */
  Vec u;                      /* solution */
  LowMachFlowContext context; /* user-defined work context */
  PetscReal t;
  PetscErrorCode ierr;

  // initialize petsc and mpi
  ierr = PetscInitialize(&argc, &args, NULL, help);
  if (ierr) return ierr;

  // setup and initialize the constant field variables
  ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(FlowParameters), &context.parameters);CHKERRQ(ierr);
  ierr = SetupFlowParameters(&context.parameters);CHKERRQ(ierr);

  // setup the ts
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm, PETSC_TRUE, 2);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &context);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  // setup problem
  ierr = SetupDiscretization(dm, &context);CHKERRQ(ierr);
  ierr = StartProblemSetup(dm, &context);CHKERRQ(ierr);

  // Override problem with source terms, boundary, and set the exact solution
  {
    PetscDS prob;
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);

    // V, W Test Function
      ierr = PetscDSSetResidual(prob, Q, f0_quadratic_q, NULL);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, V, f0_quadratic_v, VIntegrandTestGradientFunction);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, W, f0_quadratic_w, WIntegrandTestGradientFunction);CHKERRQ(ierr);

    FlowParameters *parameters;
    ierr = PetscBagGetData(context.parameters, (void **)&parameters);CHKERRQ(ierr);

    /* Setup Boundary Conditions */
    PetscInt id;
    id = 3;
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity", "marker", VEL, 0, NULL, (void (*)(void))quadratic_u, (void (*)(void))quadratic_u_t, 1, &id, parameters);CHKERRQ(ierr);
    id = 1;
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void))quadratic_u, (void (*)(void))quadratic_u_t, 1, &id, parameters);CHKERRQ(ierr);
    id = 2;
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall velocity", "marker", VEL, 0, NULL, (void (*)(void))quadratic_u, (void (*)(void))quadratic_u_t, 1, &id, parameters);CHKERRQ(ierr);
    id = 4;
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity", "marker", VEL, 0, NULL, (void (*)(void))quadratic_u, (void (*)(void))quadratic_u_t, 1, &id, parameters);CHKERRQ(ierr);
    id = 3;
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp", "marker", TEMP, 0, NULL, (void (*)(void))quadratic_T, (void (*)(void))quadratic_T_t, 1, &id, parameters);CHKERRQ(ierr);
    id = 1;
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void))quadratic_T, (void (*)(void))quadratic_T_t, 1, &id, parameters);CHKERRQ(ierr);
    id = 2;
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp", "marker", TEMP, 0, NULL, (void (*)(void))quadratic_T, (void (*)(void))quadratic_T_t, 1, &id, parameters);CHKERRQ(ierr);
    id = 4;
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp", "marker", TEMP, 0, NULL, (void (*)(void))quadratic_T, (void (*)(void))quadratic_T_t, 1, &id, parameters);CHKERRQ(ierr);

    // Set the exact solution
    ierr = PetscDSSetExactSolution(prob, VEL, quadratic_u, parameters);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob, PRES, quadratic_p, parameters);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob, TEMP, quadratic_T, parameters);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolutionTimeDerivative(prob, VEL, quadratic_u_t, parameters);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolutionTimeDerivative(prob, PRES, NULL, parameters);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolutionTimeDerivative(prob, TEMP, quadratic_T_t, parameters);CHKERRQ(ierr);
  }
  ierr = CompleteProblemSetup(ts, &u, &context);CHKERRQ(ierr);

  // Setup the TS
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  // Set initial conditions from the exact solution
  ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);CHKERRQ(ierr); /* Must come after SetFromOptions() */
  ierr = SetInitialConditions(ts, u);CHKERRQ(ierr);

  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMSetOutputSequenceNumber(dm, 0, t);CHKERRQ(ierr);
  ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts, MonitorError, &context, NULL);CHKERRQ(ierr);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)u, "Numerical Solution");CHKERRQ(ierr);
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

/*
 * -dm_plex_separate_marker -dm_refine 0  -vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1  -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1  -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged  -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full  -fieldsplit_0_pc_type lu  -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi  -ts_fd_color
 *
 */
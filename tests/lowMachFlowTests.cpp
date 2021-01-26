static char help[] =
    "Time-dependent Low Mach Flow in 2d channels with finite elements.\n\
We solve the Low Mach flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include <petsc.h>
#include "gtest/gtest.h"
#include "lowMachFlow.h"
#include "mesh.h"
#include "testFixtures/MpiTestFixture.hpp"
#include "parameters.h"

typedef PetscErrorCode (*ExactFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

struct LowMachMMSParameters {
    MpiTestParameter mpiTestParameter;
    ExactFunction uExact;
    ExactFunction pExact;
    ExactFunction TExact;
    ExactFunction u_tExact;
    ExactFunction T_tExact;
    void (*f0_q)(PetscInt,
                 PetscInt,
                 PetscInt,
                 const PetscInt[],
                 const PetscInt[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscInt[],
                 const PetscInt[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscScalar[],
                 PetscReal,
                 const PetscReal[],
                 PetscInt,
                 const PetscScalar[],
                 PetscScalar[]);
    void (*f0_v)(PetscInt,
                 PetscInt,
                 PetscInt,
                 const PetscInt[],
                 const PetscInt[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscInt[],
                 const PetscInt[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscScalar[],
                 PetscReal,
                 const PetscReal[],
                 PetscInt,
                 const PetscScalar[],
                 PetscScalar[]);
    void (*f0_w)(PetscInt,
                 PetscInt,
                 PetscInt,
                 const PetscInt[],
                 const PetscInt[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscInt[],
                 const PetscInt[],
                 const PetscScalar[],
                 const PetscScalar[],
                 const PetscScalar[],
                 PetscReal,
                 const PetscReal[],
                 PetscInt,
                 const PetscScalar[],
                 PetscScalar[]);
};

class LowMachMMS : public MpiTestFixture, public ::testing::WithParamInterface<LowMachMMSParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

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
    ierr = RemoveDiscretePressureNullspace(dm, u);
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
    T[0] = time + X[0] + X[1];
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

    f0[0] -= (Pth*S)/Power(t + x + y,2) - (4*Pth*x)/(t + x + y) +
           (Pth*(t + 2*Power(x,2) + 2*x*y))/Power(t + x + y,2) +
           (Pth*(t + Power(x,2) + Power(y,2)))/Power(t + x + y,2);
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

    f0[0] -= 1 - (5.333333333333334*mu)/R +
        (Pth*(S + (4*x + 2*y)*(t + 2*Power(x,2) + 2*x*y) + 2*x*(t + Power(x,2) + Power(y,2))))/(t + x + y);

    f0[1] -= 1 - (4.*mu)/R + Pth/(F*(t + x + y)) +
        (Pth*(S + 2*x*(t + 2*Power(x,2) + 2*x*y) + 2*y*(t + Power(x,2) + Power(y,2))))/ (t + x + y);
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

    f0[0] -= (Cp*Pth*(S + 2*t + 3*Power(x,2) + 2*x*y + Power(y,2)))/(H*S*(t + x + y));
}


TEST_P(LowMachMMS, LowMachMMSTests) {
    StartWithMPI DM dm;         /* problem definition */
        TS ts;                      /* timestepper */
        Vec u;                      /* solution */
        LowMachFlowContext context; /* user-defined work context */
        PetscReal t;
        PetscErrorCode ierr;

        // Get the testing param
        auto testingParam = GetParam();

        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, help);

        // setup and initialize the constant field variables
        ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(FlowParameters), &context.parameters);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = SetupFlowParameters(&context.parameters);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // setup the ts
        ierr = TSCreate(PETSC_COMM_WORLD, &ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = CreateMesh(PETSC_COMM_WORLD, &dm, PETSC_TRUE, 2);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSSetDM(ts, dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSetApplicationContext(dm, &context);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // setup problem
        ierr = SetupDiscretization(dm, &context);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = StartProblemSetup(dm, &context);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Override problem with source terms, boundary, and set the exact solution
        {
            PetscDS prob;
            ierr = DMGetDS(dm, &prob);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            // V, W Test Function
            ierr = PetscDSSetResidual(prob, Q, testingParam.f0_q, VIntegrandTestGradientFunction);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetResidual(prob, V, testingParam.f0_v, VIntegrandTestGradientFunction);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetResidual(prob, W, testingParam.f0_w, WIntegrandTestGradientFunction);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            FlowParameters *parameters;
            ierr = PetscBagGetData(context.parameters, (void **)&parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            /* Setup Boundary Conditions */
            PetscInt id;
            id = 3;
            ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 1;
            ierr =
                PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 2;
            ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 4;
            ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 3;
            ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 1;
            ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 2;
            ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 4;
            ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            // Set the exact solution
            ierr = PetscDSSetExactSolution(prob, VEL, testingParam.uExact, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolution(prob, PRES, testingParam.pExact, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolution(prob, TEMP, testingParam.TExact, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, VEL, testingParam.u_tExact, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, PRES, NULL, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, TEMP, testingParam.T_tExact, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        ierr = CompleteProblemSetup(ts, &u, &context);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Setup the TS
        ierr = TSSetFromOptions(ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Set initial conditions from the exact solution
        ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);
        CHKERRABORT(PETSC_COMM_WORLD, ierr); /* Must come after SetFromOptions() */
        ierr = SetInitialConditions(ts, u);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = TSGetTime(ts, &t);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSetOutputSequenceNumber(dm, 0, t);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMTSCheckFromOptions(ts, u);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSMonitorSet(ts, MonitorError, &context, NULL);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = PetscObjectSetName((PetscObject)u, "Numerical Solution");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSSolve(ts, u);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Compare the actual vs expected values
        ierr = DMTSCheckFromOptions(ts, u);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Cleanup
        ierr = VecDestroy(&u);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMDestroy(&dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSDestroy(&ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscBagDestroy(&context.parameters);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscFinalize();
        exit(ierr);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    LowMachMMSTests,
    LowMachMMS,
    testing::Values(
        (LowMachMMSParameters){
            .mpiTestParameter = {.nproc = 0,
                .expectedOutputFile = "outputs/2d_tri_p2_p1_p1",
                .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                             "-snes_fd_color "
                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                             "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                             "-fieldsplit_0_pc_type lu "
                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .uExact = quadratic_u,
            .pExact = quadratic_p,
            .TExact = quadratic_T,
            .u_tExact = quadratic_u_t,
            .T_tExact = quadratic_T_t,
            .f0_q = f0_quadratic_q,
            .f0_v = f0_quadratic_v,
            .f0_w = f0_quadratic_w}
        ));

std::ostream &operator<<(std::ostream &os, const LowMachMMSParameters &params) { return os << params.mpiTestParameter.expectedOutputFile; }
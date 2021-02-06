static char help[] =
    "Time-dependent Low Mach Flow in 2d channels with finite elements.\n\
We solve the Low Mach flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include <petsc.h>
#include "gtest/gtest.h"
#include "flow.h"
#include "mesh.h"
#include "testFixtures/MpiTestFixture.hpp"

typedef PetscErrorCode (*ExactFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

typedef void (*IntegrandTestFunction)(PetscInt dim, PetscInt Nf,PetscInt NfAux,
                                          const PetscInt *uOff,
                                          const PetscInt *uOff_x,
                                          const PetscScalar *u,
                                          const PetscScalar *u_t,
                                          const PetscScalar *u_x,
                                          const PetscInt *aOff,
                                          const PetscInt *aOff_x,
                                          const PetscScalar *a,
                                          const PetscScalar *a_t,
                                          const PetscScalar *a_x,
                                   PetscReal t,
                                          const PetscReal *X,
                                   PetscInt numConstants,
                                          const PetscScalar *constants,
                                          PetscScalar *f0);

#define SourceFunction(FUNC) FUNC(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[], PetscReal t,const PetscReal X[], PetscInt numConstants,const PetscScalar constants[], PetscScalar f0[])

// store the pointer to the provided test function from the solver
static IntegrandTestFunction f0_v_original;
static IntegrandTestFunction f0_w_original;

struct FlowMMSParameters {
    MpiTestParameter mpiTestParameter;
    FlowType flowType;
    ExactFunction uExact;
    ExactFunction pExact;
    ExactFunction TExact;
    ExactFunction u_tExact;
    ExactFunction T_tExact;
    IntegrandTestFunction f0_v;
    IntegrandTestFunction f0_w;
};

class FlowMMS : public MpiTestFixture, public ::testing::WithParamInterface<FlowMMSParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }

};

PETSC_EXTERN PetscErrorCode removeDiscretePressureNullspace(DM dm, Vec u);

static PetscErrorCode SetInitialConditions(TS ts, Vec u) {
    DM dm;
    PetscReal t;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = TSGetTime(ts, &t);CHKERRQ(ierr);

    // This function Tags the u vector as the exact solution.  We need to copy the values to prevent this.
    Vec e;
    ierr = VecDuplicate(u, &e); CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, t, e, NULL); CHKERRQ(ierr);
    ierr = VecCopy(e, u); CHKERRQ(ierr);
    ierr = VecDestroy(&e); CHKERRQ(ierr);

    // get the flow to apply the completeFlowInitialization method
    Flow flow;
    ierr = DMGetApplicationContext(dm,(void**)&flow); CHKERRQ(ierr);
    FlowInitialization(flow, dm, u);

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

// helper functions for generated code
static PetscReal Power(PetscReal x, PetscInt exp){
    return PetscPowReal(x, exp);
}
static PetscReal Cos(PetscReal x){
    return PetscCosReal(x);
}
static PetscReal Sin(PetscReal x){
    return PetscSinReal(x);
}

/*
  CASE: incompressible quadratic
  In 2D we use exact solution:

    u = t + x^2 + y^2
    v = t + 2x^2 - 2xy
    p = x + y - 1
    T = t + x + y
  so that

    \nabla \cdot u = 2x - 2x = 0

  see docs/content/formulations/incompressibleFlow/solutions/Incompressible_2D_Quadratic_MMS.nb
*/
static PetscErrorCode incompressible_quadratic_u(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = time + X[0] * X[0] + X[1] * X[1];
    u[1] = time + 2.0 * X[0] * X[0] - 2.0 * X[0] * X[1];
    return 0;
}
static PetscErrorCode incompressible_quadratic_u_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode incompressible_quadratic_p(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode incompressible_quadratic_T(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode incompressible_quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

/* f0_v = du/dt - f */
static void SourceFunction(f0_incompressible_quadratic_v){
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal mu = constants[MU];
    const PetscReal R = constants[REYNOLDS];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= 1 - (4.*mu)/R + rho*S + 2*rho*y*(t + 2*Power(x,2) - 2*x*y) + 2*rho*x*(t + Power(x,2) + Power(y,2));
    f0[1] -= 1 - (4.*mu)/R + rho*S - 2*rho*x*(t + 2*Power(x,2) - 2*x*y) + rho*(4*x - 2*y)*(t + Power(x,2) + Power(y,2));
}

/* f0_w = dT/dt + u.grad(T) - Q */
static void SourceFunction(f0_incompressible_quadratic_w){
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal Cp = constants[CP];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= Cp*rho*(S + 2*t + 3*Power(x,2) - 2*x*y + Power(y,2));
}

/*
  CASE: incompressible cubic
  In 2D we use exact solution:

    u = t + x^3 + y^3
    v = t + 2x^3 - 3x^2y
    p = 3/2 x^2 + 3/2 y^2 - 1
    T = t + 1/2 x^2 + 1/2 y^2

  so that

    \nabla \cdot u = 3x^2 - 3x^2 = 0

   see docs/content/formulations/incompressibleFlow/solutions/Incompressible_2D_Cubic_MMS.nb.nb
*/
static PetscErrorCode incompressible_cubic_u(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = time + X[0] * X[0] * X[0] + X[1] * X[1] * X[1];
    u[1] = time + 2.0 * X[0] * X[0] * X[0] - 3.0 * X[0] * X[0] * X[1];
    return 0;
}
static PetscErrorCode incompressible_cubic_u_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode incompressible_cubic_p(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 3.0 * X[0] * X[0] / 2.0 + 3.0 * X[1] * X[1] / 2.0 - 1.0;
    return 0;
}

static PetscErrorCode incompressible_cubic_T(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] * X[0] / 2.0 + X[1] * X[1] / 2.0;
    return 0;
}
static PetscErrorCode incompressible_cubic_T_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

static void SourceFunction(f0_incompressible_cubic_v){
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal mu = constants[MU];
    const PetscReal R = constants[REYNOLDS];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= rho*S + 3*x + 3*rho*Power(y,2)*(t + 2*Power(x,3) - 3*Power(x,2)*y) + 3*rho*Power(x,2)*(t + Power(x,3) + Power(y,3)) - (12.*mu*x + 1.*mu*(-6*x + 6*y))/R;
    f0[1] -= rho*S - (1.*mu*(12*x - 6*y))/R + 3*y - 3*rho*Power(x,2)*(t + 2*Power(x,3) - 3*Power(x,2)*y) + rho*(6*Power(x,2) - 6*x*y)*(t + Power(x,3) + Power(y,3));
}

static void SourceFunction(f0_incompressible_cubic_w){
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal Cp = constants[CP];
    const PetscReal k = constants[K];
    const PetscReal P = constants[PECLET];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= (-2.*k)/P + Cp*rho*(S + 1.*y*(t + 2*Power(x,3) - 3*Power(x,2)*y) + 1.*x*(t + Power(x,3) + Power(y,3)));
}

/*
  CASE: incompressible cubic-trigonometric
  In 2D we use exact solution:

    u = beta cos t + x^3 + y^3
    v = beta sin t + 2x^3 - 3x^2y
    p = 3/2 x^2 + 3/2 y^2 - 1
    T = 20 cos t + 1/2 x^2 + 1/2 y^2
    f = < beta cos t 3x^2         + beta sin t (3y^2 - 1) + 3x^5 + 6x^3y^2 - 6x^2y^3 - \nu(6x + 6y)  + 3x,
          beta cos t (6x^2 - 6xy) - beta sin t (3x^2)     + 3x^4y + 6x^2y^3 - 6xy^4  - \nu(12x - 6y) + 3y>
    Q = beta cos t x + beta sin t (y - 1) + x^4 + 2x^3y - 3x^2y^2 + xy^3 - 2\alpha

  so that

    \nabla \cdot u = 3x^2 - 3x^2 = 0

   see docs/content/formulations/incompressibleFlow/solutions/Incompressible_2D_Cubic-Trigonometric_MMS.nb.nb

*/
static PetscErrorCode incompressible_cubic_trig_u(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    const PetscReal beta = 100.0;
    u[0] = beta * Cos(time) + X[0] * X[0] * X[0] + X[1] * X[1] * X[1];
    u[1] = beta * Sin(time) + 2.0 * X[0] * X[0] * X[0] - 3.0 * X[0] * X[0] * X[1];
    return 0;
}
static PetscErrorCode incompressible_cubic_trig_u_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *u, void *ctx) {
    const PetscReal beta = 100.0;
    u[0] = -beta * Sin(time);
    u[1] = beta * Cos(time);
    return 0;
}

static PetscErrorCode incompressible_cubic_trig_p(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 3.0 * X[0] * X[0] / 2.0 + 3.0 * X[1] * X[1] / 2.0 - 1.0;
    return 0;
}

static PetscErrorCode incompressible_cubic_trig_T(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    const PetscReal beta = 100.0;
    T[0] = beta * Cos(time) + X[0] * X[0] / 2.0 + X[1] * X[1] / 2.0;
    return 0;
}
static PetscErrorCode incompressible_cubic_trig_T_t(PetscInt Dim, PetscReal time, const PetscReal *X, PetscInt Nf, PetscScalar *T, void *ctx) {
    const PetscReal beta = 100.0;
    T[0] = -beta * Sin(time);
    return 0;
}

static void SourceFunction(f0_incompressible_cubic_trig_v){
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal beta = 100.0;
    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal mu = constants[MU];
    const PetscReal R = constants[REYNOLDS];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= 3*x - (12.*mu*x + 1.*mu*(-6*x + 6*y))/R + 3*rho*Power(x,2)*(Power(x,3) + Power(y,3) + beta*Cos(t)) - beta*rho*S*Sin(t) + 3*rho*Power(y,2)*(2*Power(x,3) - 3*Power(x,2)*y + beta*Sin(t));
    f0[1] -= (-1.*mu*(12*x - 6*y))/R + 3*y + beta*rho*S*Cos(t) + rho*(6*Power(x,2) - 6*x*y)*(Power(x,3) + Power(y,3) + beta*Cos(t)) - 3*rho*Power(x,2)*(2*Power(x,3) - 3*Power(x,2)*y + beta*Sin(t));
}

static void SourceFunction(f0_incompressible_cubic_trig_w){
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal beta = 100.0;
    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal Cp = constants[CP];
    const PetscReal k = constants[K];
    const PetscReal P = constants[PECLET];
    const PetscReal x = X[0];
    const PetscReal y = X[1];

    f0[0] -= (-2*k)/P + Cp*rho*(x*(Power(x,3) + Power(y,3) + beta*Cos(t)) - beta*S*Sin(t) + y*(2*Power(x,3) - 3*Power(x,2)*y + beta*Sin(t)));
}

TEST_P(FlowMMS, MachMMSTests) {
    StartWithMPI DM dm;         /* problem definition */
        TS ts;                      /* timestepper */
        Flow flow; /* user-defined work context */
        PetscReal t;
        PetscErrorCode ierr;

        // Get the testing param
        auto testingParam = GetParam();

        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, help);

        // setup the ts
        ierr = TSCreate(PETSC_COMM_WORLD, &ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = CreateMesh(PETSC_COMM_WORLD, &dm, PETSC_TRUE, 2);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSSetDM(ts, dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // create a flow
        ierr = FlowCreate(&flow,testingParam.flowType, dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);


        // setup problem
        ierr = FlowSetupDiscretization(flow);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = FlowStartProblemSetup(flow);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Override problem with source terms, boundary, and set the exact solution
        {
            PetscDS prob;
            ierr = DMGetDS(dm, &prob);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            // V, W Test Function
            IntegrandTestFunction tempFunctionPointer;
            ierr = PetscDSGetResidual(prob, V, &f0_v_original, &tempFunctionPointer);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetResidual(prob, V, testingParam.f0_v, tempFunctionPointer);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSGetResidual(prob, W, &f0_w_original, &tempFunctionPointer);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetResidual(prob, W, testingParam.f0_w, tempFunctionPointer);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            FlowParameters *parameters;
            ierr = PetscBagGetData(flow->parameters, (void **)&parameters);
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
        ierr = FlowCompleteProblemSetup(flow, ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Setup the TS
        ierr = TSSetFromOptions(ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Set initial conditions from the exact solution
        ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);
        CHKERRABORT(PETSC_COMM_WORLD, ierr); /* Must come after SetFromOptions() */
        ierr = SetInitialConditions(ts, flow->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = TSGetTime(ts, &t);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSetOutputSequenceNumber(dm, 0, t);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMTSCheckFromOptions(ts, flow->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSMonitorSet(ts, MonitorError, flow, NULL);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = TSSolve(ts, flow->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Compare the actual vs expected values
        ierr = DMTSCheckFromOptions(ts, flow->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Cleanup
        ierr = DMDestroy(&dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSDestroy(&ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = FlowDestroy(&flow);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscFinalize();
        exit(ierr);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    FlowMMSTests,
    FlowMMS,
    testing::Values(
        (FlowMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d quadratic tri_p2_p1_p1",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/incompressible_2d_tri_p2_p1_p1",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                                              "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = incompressible_quadratic_u,
            .pExact = incompressible_quadratic_p,
            .TExact = incompressible_quadratic_T,
            .u_tExact = incompressible_quadratic_u_t,
            .T_tExact = incompressible_quadratic_T_t,
            .f0_v = f0_incompressible_quadratic_v,
            .f0_w = f0_incompressible_quadratic_w},
        (FlowMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d quadratic tri_p2_p1_p1 4 proc",
                                 .nproc = 4,
                                 .expectedOutputFile = "outputs/incompressible_2d_tri_p2_p1_p1_nproc4",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 1 -dm_distribute "
                                              "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = incompressible_quadratic_u,
            .pExact = incompressible_quadratic_p,
            .TExact = incompressible_quadratic_T,
            .u_tExact = incompressible_quadratic_u_t,
            .T_tExact = incompressible_quadratic_T_t,
            .f0_v = f0_incompressible_quadratic_v,
            .f0_w = f0_incompressible_quadratic_w},
        (FlowMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d cubic trig tri_p2_p1_p1_tconv",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/incompressible_2d_tri_p2_p1_p1_tconv",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                                              "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                              "-ts_max_steps 4 -ts_dt 0.1 -ts_convergence_estimate -convest_num_refine 1 "
                                              "-snes_error_if_not_converged -snes_convergence_test correct_pressure "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = incompressible_cubic_trig_u,
            .pExact = incompressible_cubic_trig_p,
            .TExact = incompressible_cubic_trig_T,
            .u_tExact = incompressible_cubic_trig_u_t,
            .T_tExact = incompressible_cubic_trig_T_t,
            .f0_v = f0_incompressible_cubic_trig_v,
            .f0_w = f0_incompressible_cubic_trig_w},
        (FlowMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d cubic p2_p1_p1_sconv",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/incompressible_2d_tri_p2_p1_p1_sconv",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                                              "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                              "-ts_max_steps 1 -ts_dt 1e-4 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 "
                                              "-snes_error_if_not_converged -snes_convergence_test correct_pressure "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_atol 1e-16 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = incompressible_cubic_u,
            .pExact = incompressible_cubic_p,
            .TExact = incompressible_cubic_T,
            .u_tExact = incompressible_cubic_u_t,
            .T_tExact = incompressible_cubic_T_t,
            .f0_v = f0_incompressible_cubic_v,
            .f0_w = f0_incompressible_cubic_w},
        (FlowMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d cubic tri_p3_p2_p2",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/incompressible_2d_tri_p3_p2_p2",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                                              "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                              "-snes_convergence_test correct_pressure "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = incompressible_cubic_u,
            .pExact = incompressible_cubic_p,
            .TExact = incompressible_cubic_T,
            .u_tExact = incompressible_cubic_u_t,
            .T_tExact = incompressible_cubic_T_t,
            .f0_v = f0_incompressible_cubic_v,
            .f0_w = f0_incompressible_cubic_w},
        (FlowMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d quadratic tri_p2_p1_p1 with real coefficients",
                .nproc = 1,
                .expectedOutputFile = "outputs/incompressible_2d_tri_p2_p1_p1_real_coefficients",
                .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                             "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                             "-fieldsplit_0_pc_type lu "
                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                             "-pth 91282.5 -strouhal 0.00242007695844728 -reynolds 23126.2780617827 -froude 0.316227766016838 -peclet 16373.1785965753 "
                             "-heatRelease 0.00831162126672484 -gamma 0.285337972166998 -mu 1.1 -k 1.2 -cp 1.3 "},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = incompressible_quadratic_u,
            .pExact = incompressible_quadratic_p,
            .TExact = incompressible_quadratic_T,
            .u_tExact = incompressible_quadratic_u_t,
            .T_tExact = incompressible_quadratic_T_t,
            .f0_v = f0_incompressible_quadratic_v,
            .f0_w = f0_incompressible_quadratic_w},
        (FlowMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d cubic tri_p3_p2_p2 with real coefficients",
                .nproc = 1,
                .expectedOutputFile = "outputs/incompressible_2d_tri_p3_p2_p2_real_coefficients",
                .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                             "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                             "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                             "-snes_convergence_test correct_pressure "
                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                             "-fieldsplit_0_pc_type lu "
                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                             "-pth 91282.5 -strouhal 0.00242007695844728 -reynolds 23126.2780617827 -froude 0.316227766016838 -peclet 16373.1785965753 "
                             "-heatRelease 0.00831162126672484 -gamma 0.285337972166998 -mu 1.1 -k 1.2 -cp 1.3 "},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = incompressible_cubic_u,
            .pExact = incompressible_cubic_p,
            .TExact = incompressible_cubic_T,
            .u_tExact = incompressible_cubic_u_t,
            .T_tExact = incompressible_cubic_T_t,
            .f0_v = f0_incompressible_cubic_v,
            .f0_w = f0_incompressible_cubic_w}));

std::ostream &operator<<(std::ostream &os, const FlowMMSParameters &params) { return os << params.mpiTestParameter; }
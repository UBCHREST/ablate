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

/*
  CASE: quadratic
  In 2D we use exact solution:

    u = t + x^2 + y^2
    v = t + 2x^2 - 2xy
    p = x + y - 1
    T = t + x + y
    f = <t (2x + 2y) + 2x^3 + 4x^2y - 2xy^2 -4\nu + 2, t (2x - 2y) + 4xy^2 + 2x^2y - 2y^3 -4\nu + 2>
    Q = 1 + 2t + 3x^2 - 2xy + y^2

  so that

    \nabla \cdot u = 2x - 2x = 0

  f = du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p
    = <1, 1> + <t + x^2 + y^2, t + 2x^2 - 2xy> . <<2x, 4x - 2y>, <2y, -2x>> - \nu <4, 4> + <1, 1>
    = <t (2x + 2y) + 2x^3 + 4x^2y - 2xy^2, t (2x - 2y) + 2x^2y + 4xy^2 - 2y^3> + <-4 \nu + 2, -4\nu + 2>
    = <t (2x + 2y) + 2x^3 + 4x^2y - 2xy^2 - 4\nu + 2, t (2x - 2y) + 4xy^2 + 2x^2y - 2y^3 - 4\nu + 2>

  Q = dT/dt + u \cdot \nabla T - \alpha \Delta T
    = 1 + <t + x^2 + y^2, t + 2x^2 - 2xy> . <1, 1> - \alpha 0
    = 1 + 2t + 3x^2 - 2xy + y^2
*/
static PetscErrorCode quadratic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = time + X[0] * X[0] + X[1] * X[1];
    u[1] = time + 2.0 * X[0] * X[0] - 2.0 * X[0] * X[1];
    return 0;
}
static PetscErrorCode quadratic_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode quadratic_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode quadratic_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode quadratic_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

/* f0_v = du/dt - f */
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
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    const PetscReal nu = PetscRealPart(1.0);

    f0[0] -= (t * (2 * X[0] + 2 * X[1]) + 2 * X[0] * X[0] * X[0] + 4 * X[0] * X[0] * X[1] - 2 * X[0] * X[1] * X[1] - 4.0 * nu + 2);
    f0[1] -= (t * (2 * X[0] - 2 * X[1]) + 4 * X[0] * X[1] * X[1] + 2 * X[0] * X[0] * X[1] - 2 * X[1] * X[1] * X[1] - 4.0 * nu + 2);
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
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] += -(2 * t + 1 + 3 * X[0] * X[0] - 2 * X[0] * X[1] + X[1] * X[1]);
}

/*
  CASE: cubic
  In 2D we use exact solution:

    u = t + x^3 + y^3
    v = t + 2x^3 - 3x^2y
    p = 3/2 x^2 + 3/2 y^2 - 1
    T = t + 1/2 x^2 + 1/2 y^2
    f = < t(3x^2 + 3y^2) + 3x^5 + 6x^3y^2 - 6x^2y^3 - \nu(6x + 6y) + 3x + 1,
          t(3x^2 - 6xy) + 6x^2y^3 + 3x^4y - 6xy^4 - \nu(12x - 6y) + 3y + 1>
    Q = x^4 + xy^3 + 2x^3y - 3x^2y^2 + xt + yt - 2\alpha + 1

  so that

    \nabla \cdot u = 3x^2 - 3x^2 = 0

  du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p - f
  = <1,1> + <t(3x^2 + 3y^2) + 3x^5 + 6x^3y^2 - 6x^2y^3, t(3x^2 - 6xy) + 6x^2y^3 + 3x^4y - 6xy^4> - \nu<6x + 6y, 12x - 6y> + <3x, 3y> - <t(3x^2 + 3y^2) + 3x^5 + 6x^3y^2 - 6x^2y^3 - \nu(6x + 6y) + 3x +
  1, t(3x^2 - 6xy) + 6x^2y^3 + 3x^4y - 6xy^4 - \nu(12x - 6y) + 3y + 1>  = 0

  dT/dt + u \cdot \nabla T - \alpha \Delta T - Q = 1 + (x^3 + y^3) x + (2x^3 - 3x^2y) y - 2*\alpha - (x^4 + xy^3 + 2x^3y - 3x^2y^2 - 2*\alpha +1)   = 0
*/
static PetscErrorCode cubic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = time + X[0] * X[0] * X[0] + X[1] * X[1] * X[1];
    u[1] = time + 2.0 * X[0] * X[0] * X[0] - 3.0 * X[0] * X[0] * X[1];
    return 0;
}
static PetscErrorCode cubic_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = 1.0;
    return 0;
}

static PetscErrorCode cubic_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 3.0 * X[0] * X[0] / 2.0 + 3.0 * X[1] * X[1] / 2.0 - 1.0;
    return 0;
}

static PetscErrorCode cubic_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] * X[0] / 2.0 + X[1] * X[1] / 2.0;
    return 0;
}
static PetscErrorCode cubic_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

static void f0_cubic_v(PetscInt dim,
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
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal nu = PetscRealPart(1.0);
    f0[0] -= (t * (3 * X[0] * X[0] + 3 * X[1] * X[1]) + 3 * X[0] * X[0] * X[0] * X[0] * X[0] + 6 * X[0] * X[0] * X[0] * X[1] * X[1] - 6 * X[0] * X[0] * X[1] * X[1] * X[1] -
              (6 * X[0] + 6 * X[1]) * nu + 3 * X[0] + 1);
    f0[1] -= (t * (3 * X[0] * X[0] - 6 * X[0] * X[1]) + 3 * X[0] * X[0] * X[0] * X[0] * X[1] + 6 * X[0] * X[0] * X[1] * X[1] * X[1] - 6 * X[0] * X[1] * X[1] * X[1] * X[1] -
              (12 * X[0] - 6 * X[1]) * nu + 3 * X[1] + 1);
}

static void f0_cubic_w(PetscInt dim,
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
    const PetscReal alpha = PetscRealPart(constants[1]);
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] += -(X[0] * X[0] * X[0] * X[0] + 2.0 * X[0] * X[0] * X[0] * X[1] - 3.0 * X[0] * X[0] * X[1] * X[1] + X[0] * X[1] * X[1] * X[1] + X[0] * t + X[1] * t - 2.0 * alpha + 1);
}

/*
  CASE: cubic-trigonometric
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

  f = du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p
    = <-sin t, cos t> + <cos t + x^3 + y^3, sin t + 2x^3 - 3x^2y> <<3x^2, 6x^2 - 6xy>, <3y^2, -3x^2>> - \nu <6x + 6y, 12x - 6y> + <3x, 3y>
    = <-sin t, cos t> + <cos t 3x^2 + 3x^5 + 3x^2y^3 + sin t 3y^2 + 6x^3y^2 - 9x^2y^3, cos t (6x^2 - 6xy) + 6x^5 - 6x^4y + 6x^2y^3 - 6xy^4 + sin t (-3x^2) - 6x^5 + 9x^4y> - \nu <6x + 6y, 12x - 6y> +
  <3x, 3y> = <cos t (3x^2)       + sin t (3y^2 - 1) + 3x^5 + 6x^3y^2 - 6x^2y^3 - \nu (6x + 6y)  + 3x, cos t (6x^2 - 6xy) - sin t (3x^2)     + 3x^4y + 6x^2y^3 - 6xy^4  - \nu (12x - 6y) + 3y>

  Q = dT/dt + u \cdot \nabla T - \alpha \Delta T
    = -sin t + <cos t + x^3 + y^3, sin t + 2x^3 - 3x^2y> . <x, y> - 2 \alpha
    = -sin t + cos t (x) + x^4 + xy^3 + sin t (y) + 2x^3y - 3x^2y^2 - 2 \alpha
    = cos t x + sin t (y - 1) + (x^4 + 2x^3y - 3x^2y^2 + xy^3 - 2 \alpha)
*/
static PetscErrorCode cubic_trig_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 100. * PetscCosReal(time) + X[0] * X[0] * X[0] + X[1] * X[1] * X[1];
    u[1] = 100. * PetscSinReal(time) + 2.0 * X[0] * X[0] * X[0] - 3.0 * X[0] * X[0] * X[1];
    return 0;
}
static PetscErrorCode cubic_trig_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = -100. * PetscSinReal(time);
    u[1] = 100. * PetscCosReal(time);
    return 0;
}

static PetscErrorCode cubic_trig_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 3.0 * X[0] * X[0] / 2.0 + 3.0 * X[1] * X[1] / 2.0 - 1.0;
    return 0;
}

static PetscErrorCode cubic_trig_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 100. * PetscCosReal(time) + X[0] * X[0] / 2.0 + X[1] * X[1] / 2.0;
    return 0;
}
static PetscErrorCode cubic_trig_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = -100. * PetscSinReal(time);
    return 0;
}

static void f0_cubic_trig_v(PetscInt dim,
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
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal nu = PetscRealPart(1.0);
    f0[0] -= 100. * PetscCosReal(t) * (3 * X[0] * X[0]) + 100. * PetscSinReal(t) * (3 * X[1] * X[1] - 1.) + 3 * X[0] * X[0] * X[0] * X[0] * X[0] + 6 * X[0] * X[0] * X[0] * X[1] * X[1] -
             6 * X[0] * X[0] * X[1] * X[1] * X[1] - (6 * X[0] + 6 * X[1]) * nu + 3 * X[0];
    f0[1] -= 100. * PetscCosReal(t) * (6 * X[0] * X[0] - 6 * X[0] * X[1]) - 100. * PetscSinReal(t) * (3 * X[0] * X[0]) + 3 * X[0] * X[0] * X[0] * X[0] * X[1] + 6 * X[0] * X[0] * X[1] * X[1] * X[1] -
             6 * X[0] * X[1] * X[1] * X[1] * X[1] - (12 * X[0] - 6 * X[1]) * nu + 3 * X[1];
}

static void f0_cubic_trig_w(PetscInt dim,
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
    const PetscReal alpha = PetscRealPart(constants[1]);
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] += -(100. * PetscCosReal(t) * X[0] + 100. * PetscSinReal(t) * (X[1] - 1.) + X[0] * X[0] * X[0] * X[0] + 2.0 * X[0] * X[0] * X[0] * X[1] - 3.0 * X[0] * X[0] * X[1] * X[1] +
               X[0] * X[1] * X[1] * X[1] - 2.0 * alpha);
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
            .mpiTestParameter = {
                .testName = "incompressible 2d quadratic tri_p2_p1_p1",
                .nproc = 1,
                .expectedOutputFile = "outputs/2d_tri_p2_p1_p1",
                .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                             "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                             "-fieldsplit_0_pc_type lu "
                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = quadratic_u,
            .pExact = quadratic_p,
            .TExact = quadratic_T,
            .u_tExact = quadratic_u_t,
            .T_tExact = quadratic_T_t,
            .f0_v = f0_quadratic_v,
            .f0_w = f0_quadratic_w},
        (FlowMMSParameters){
            .mpiTestParameter = {
                .testName = "incompressible 2d quadratic tri_p2_p1_p1 4 proc",
                .nproc = 4,
                .expectedOutputFile = "outputs/2d_tri_p2_p1_p1_nproc4",
                .arguments = "-dm_plex_separate_marker -dm_refine 1 -dm_distribute "
                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                             "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                             "-fieldsplit_0_pc_type lu "
                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = quadratic_u,
            .pExact = quadratic_p,
            .TExact = quadratic_T,
            .u_tExact = quadratic_u_t,
            .T_tExact = quadratic_T_t,
            .f0_v = f0_quadratic_v,
            .f0_w = f0_quadratic_w},
        (FlowMMSParameters){
            .mpiTestParameter = {
                .testName = "incompressible 2d cubic trig tri_p2_p1_p1_tconv 4 proc",
                .nproc = 1,
                .expectedOutputFile = "outputs/2d_tri_p2_p1_p1_tconv",
                .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                             "-ts_max_steps 4 -ts_dt 0.1 -ts_convergence_estimate -convest_num_refine 1 "
                             "-snes_error_if_not_converged -snes_convergence_test correct_pressure "
                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                             "-fieldsplit_0_pc_type lu "
                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = cubic_trig_u,
            .pExact = cubic_trig_p,
            .TExact = cubic_trig_T,
            .u_tExact = cubic_trig_u_t,
            .T_tExact = cubic_trig_T_t,
            .f0_v = f0_cubic_trig_v,
            .f0_w = f0_cubic_trig_w},
        (FlowMMSParameters){
            .mpiTestParameter = {
                .testName = "incompressible 2d cubic p2_p1_p1_sconv",
                .nproc = 1,
                .expectedOutputFile = "outputs/2d_tri_p2_p1_p1_sconv",
                .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                             "-ts_max_steps 1 -ts_dt 1e-4 -ts_convergence_estimate -ts_convergence_temporal 0 -convest_num_refine 1 "
                             "-snes_error_if_not_converged -snes_convergence_test correct_pressure "
                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_atol 1e-16 -ksp_error_if_not_converged "
                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                             "-fieldsplit_0_pc_type lu "
                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = cubic_u,
            .pExact = cubic_p,
            .TExact = cubic_T,
            .u_tExact = cubic_u_t,
            .T_tExact = cubic_T_t,
            .f0_v = f0_cubic_v,
            .f0_w = f0_cubic_w},
        (FlowMMSParameters){
            .mpiTestParameter = {
                .testName = "incompressible 2d cubic tri_p3_p2_p2",
                .nproc = 1,
                .expectedOutputFile = "outputs/2d_tri_p3_p2_p2",
                .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                             "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                             "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                             "-snes_convergence_test correct_pressure "
                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                             "-fieldsplit_0_pc_type lu "
                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"},
            .flowType = FLOWINCOMPRESSIBLE,
            .uExact = cubic_u,
            .pExact = cubic_p,
            .TExact = cubic_T,
            .u_tExact = cubic_u_t,
            .T_tExact = cubic_T_t,
            .f0_v = f0_cubic_v,
            .f0_w = f0_cubic_w}));

std::ostream &operator<<(std::ostream &os, const FlowMMSParameters &params) { return os << params.mpiTestParameter; }
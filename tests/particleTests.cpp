#include <petsc.h>
#include "flow.h"
#include "gtest/gtest.h"
#include "mesh.h"
#include "particleInitializer.h"
#include "particles.h"
#include "testFixtures/MpiTestFixture.hpp"

typedef PetscErrorCode (*ExactFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

typedef void (*IntegrandTestFunction)(PetscInt dim,
                                      PetscInt Nf,
                                      PetscInt NfAux,
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

#define SourceFunction(FUNC)            \
    FUNC(PetscInt dim,                  \
         PetscInt Nf,                   \
         PetscInt NfAux,                \
         const PetscInt uOff[],         \
         const PetscInt uOff_x[],       \
         const PetscScalar u[],         \
         const PetscScalar u_t[],       \
         const PetscScalar u_x[],       \
         const PetscInt aOff[],         \
         const PetscInt aOff_x[],       \
         const PetscScalar a[],         \
         const PetscScalar a_t[],       \
         const PetscScalar a_x[],       \
         PetscReal t,                   \
         const PetscReal X[],           \
         PetscInt numConstants,         \
         const PetscScalar constants[], \
         PetscScalar f0[])

// store the pointer to the provided test function from the solver
static IntegrandTestFunction f0_v_original;
static IntegrandTestFunction f0_w_original;
static IntegrandTestFunction f0_q_original;
static PetscReal omega;

struct ParticleMMSParameters {
    MpiTestParameter mpiTestParameter;
    FlowType flowType;
    ExactFunction uExact;
    ExactFunction pExact;
    ExactFunction TExact;
    ExactFunction u_tExact;
    ExactFunction T_tExact;
    ExactFunction particleExact;
    IntegrandTestFunction f0_v;
    IntegrandTestFunction f0_w;
    IntegrandTestFunction f0_q;
    PetscReal omega;
};

class ParticleMMS : public MpiTestFixture, public ::testing::WithParamInterface<ParticleMMSParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

/*
  CASE: trigonometric-trigonometric
  In 2D we use exact solution:

    x = r0 cos(w t + theta0)  r0     = sqrt(x0^2 + y0^2)
    y = r0 sin(w t + theta0)  theta0 = arctan(y0/x0)
    u = -w r0 sin(theta0) = -w y
    v =  w r0 cos(theta0) =  w x
    p = x + y - 1
    T = t + x + y
    f = <1, 1>
    Q = 1 + w (x - y)/r

  so that

    \nabla \cdot u = 0 + 0 = 0

  f = du/dt + u \cdot \nabla u - \nu \Delta u + \nabla p
    = <0, 0> + u_i d_i u_j - \nu 0 + <1, 1>
    = <1, 1> + w^2 <-y, x> . <<0, 1>, <-1, 0>>
    = <1, 1> + w^2 <-x, -y>
    = <1, 1> - w^2 <x, y>

  Q = dT/dt + u \cdot \nabla T - \alpha \Delta T
    = 1 + <u, v> . <1, 1> - \alpha 0
    = 1 + u + v
*/
static PetscErrorCode trig_trig_x(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *x, void *ctx) {
    const PetscReal x0 = X[0];
    const PetscReal y0 = X[1];
    const PetscReal R0 = PetscSqrtReal(x0 * x0 + y0 * y0);
    const PetscReal theta0 = PetscAtan2Real(y0, x0);

    x[0] = R0 * PetscCosReal(omega * time + theta0);
    x[1] = R0 * PetscSinReal(omega * time + theta0);
    return 0;
}
static PetscErrorCode trig_trig_u(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = -omega * X[1];
    u[1] = omega * X[0];
    return 0;
}
static PetscErrorCode trig_trig_u_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 0.0;
    u[1] = 0.0;
    return 0;
}

static PetscErrorCode trig_trig_p(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode trig_trig_T(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode trig_trig_T_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

static void SourceFunction(f0_trig_trig_v) {
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] -= 1.0 - omega * omega * X[0];
    f0[1] -= 1.0 - omega * omega * X[1];
}

static void SourceFunction(f0_trig_trig_w) {
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    f0[0] += -(1.0 + omega * (X[0] - X[1]));
}

static PetscErrorCode SetInitialConditions(TS ts, Vec u) {
    DM dm;
    PetscReal t;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = TSGetTime(ts, &t);
    CHKERRQ(ierr);

    // This function Tags the u vector as the exact solution.  We need to copy the values to prevent this.
    Vec e;
    ierr = VecDuplicate(u, &e);
    CHKERRQ(ierr);
    ierr = DMComputeExactSolution(dm, t, e, NULL);
    CHKERRQ(ierr);
    ierr = VecCopy(e, u);
    CHKERRQ(ierr);
    ierr = VecDestroy(&e);
    CHKERRQ(ierr);

    // get the flow to apply the completeFlowInitialization method
    Flow flow;
    ierr = DMGetApplicationContext(dm, (void **)&flow);
    CHKERRQ(ierr);
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

TEST_P(ParticleMMS, ParticleFlowMMSTests) {
    StartWithMPI
        DM dm;               /* problem definition */
        TS ts;               /* timestepper */
        Flow flow;           /* user-defined flow context */
        Particles particles; /* user-defined particle context */
        PetscReal t;
        PetscErrorCode ierr;

        // Get the testing param
        auto testingParam = GetParam();
        omega = testingParam.omega;

        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, NULL);

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
        ierr = FlowCreate(&flow, testingParam.flowType, dm);
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
            if (testingParam.f0_v) {
                ierr = PetscDSGetResidual(prob, V, &f0_v_original, &tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
                ierr = PetscDSSetResidual(prob, V, testingParam.f0_v, tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
            }
            if (testingParam.f0_w) {
                ierr = PetscDSGetResidual(prob, W, &f0_w_original, &tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
                ierr = PetscDSSetResidual(prob, W, testingParam.f0_w, tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
            }
            if (testingParam.f0_q) {
                ierr = PetscDSGetResidual(prob, Q, &f0_q_original, &tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
                ierr = PetscDSSetResidual(prob, Q, testingParam.f0_q, tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
            }

            FlowParameters *parameters;
            ierr = PetscBagGetData(flow->parameters, (void **)&parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            /* Setup Boundary Conditions */
            PetscInt id;
            id = 3;
            ierr =
                PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 1;
            ierr = PetscDSAddBoundary(
                prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 2;
            ierr = PetscDSAddBoundary(
                prob, DM_BC_ESSENTIAL, "right wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 4;
            ierr =
                PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 3;
            ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 1;
            ierr =
                PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameters);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 2;
            ierr =
                PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameters);
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
        // Setup the TS
        ierr = TSSetFromOptions(ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Initialize the particles
        ParticleInitializer particleInitializer;
        ierr = ParticleInitializerCreate(&particleInitializer);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Setup the particle domain
        ierr = ParticleCreate(&particles, PARTICLETRACER, flow, particleInitializer);
        ParticleSetExactSolution(particles, testingParam.particleExact);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Setup particle position integrator
        TS particleTs;
        ierr = TSCreate(PETSC_COMM_WORLD, &particleTs);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = ParticleSetupIntegrator(particles, particleTs, ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Solve the one way coupled system
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
        ierr = TSDestroy(&particleTs);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = FlowDestroy(&flow);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = ParticleDestroy(&particles);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = ParticleInitializerDestroy(&particleInitializer);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscFinalize();
        exit(ierr);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(ParticleMMSTests,
                         ParticleMMS,
                         testing::Values((ParticleMMSParameters){.mpiTestParameter = {.testName = "particle in incompressible 2d trigonometric-trigonometric tri_p2_p1_p1",
                                                                                      .nproc = 1,
                                                                                      .expectedOutputFile = "outputs/particle_incompressible_trigonometric_2d_tri_p2_p1_p1",
                                                                                      .arguments = "-dm_plex_separate_marker -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 "
                                                                                                   "-temp_petscspace_degree 1 -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 -ts_monitor_cancel "
                                                                                                   "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                                                                                   "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 "
                                                                                                   "-pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_0_pc_type lu "
                                                                                                   "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -particle_layout_type box "
                                                                                                   "-particle_lower 0.25,0.25 -particle_upper 0.75,0.75 -Npb 5 -particle_ts_max_steps 2 "
                                                                                                   "-particle_ts_dt 0.05 -particle_ts_convergence_estimate -convest_num_refine 1 "
                                                                                                   "-particle_ts_monitor_cancel"},
                                                                 .flowType = FLOWINCOMPRESSIBLE,
                                                                 .uExact = trig_trig_u,
                                                                 .pExact = trig_trig_p,
                                                                 .TExact = trig_trig_T,
                                                                 .u_tExact = trig_trig_u_t,
                                                                 .T_tExact = trig_trig_T_t,
                                                                 .particleExact = trig_trig_x,
                                                                 .f0_v = f0_trig_trig_v,
                                                                 .f0_w = f0_trig_trig_w,
                                                                 .omega = 0.5}));

std::ostream &operator<<(std::ostream &os, const ParticleMMSParameters &params) { return os << params.mpiTestParameter; }
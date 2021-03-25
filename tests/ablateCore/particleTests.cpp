#include <inttypes.h>
#include <petsc.h>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"
#include "incompressibleFlow.h"
#include "mesh.h"
#include "particleInitializer.h"
#include "particleTracer.h"
#include "particles.h"
typedef PetscErrorCode (*ExactFunction)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

typedef void (*IntegrandTestFunction)(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *u, const PetscScalar *u_t, const PetscScalar *u_x,
                                      const PetscInt *aOff, const PetscInt *aOff_x, const PetscScalar *a, const PetscScalar *a_t, const PetscScalar *a_x, PetscReal t, const PetscReal *X,
                                      PetscInt numConstants, const PetscScalar *constants, PetscScalar *f0);

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
    testingResources::MpiTestParameter mpiTestParameter;
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

class ParticleMMS : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<ParticleMMSParameters> {
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

/*
  CASE: linear particle movement
  In 2D we use exact solution:

    x = t + xo
    y = t*t/2 + t*xo + yo
    u = 1
    v = x
    p = x + y - 1
    T = t + x + y

  so that

    \nabla \cdot u = 0 + 0 = 0

  // see docs/content/formulations/incompressibleFlow/solutions/Incompressible_2D_Linear_MMS.nb
*/
static PetscErrorCode linear_x(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *x, void *ctx) {
    const PetscReal x0 = X[0];
    const PetscReal y0 = X[1];

    x[0] = time + x0;
    x[1] = time * time / 2 + time * x0 + y0;
    return 0;
}
static PetscErrorCode linear_u(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 1.0;
    u[1] = X[0];
    return 0;
}
static PetscErrorCode linear_u_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    u[0] = 0.0;
    u[1] = 0.0;
    return 0;
}

static PetscErrorCode linear_p(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = X[0] + X[1] - 1.0;
    return 0;
}

static PetscErrorCode linear_T(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = time + X[0] + X[1];
    return 0;
}
static PetscErrorCode linear_T_t(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 1.0;
    return 0;
}

static void SourceFunction(f0_linear_v) {
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal rho = 1.0;

    f0[0] -= 1;
    f0[1] -= 1 + rho;
}

static void SourceFunction(f0_linear_w) {
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);

    const PetscReal rho = 1.0;
    const PetscReal S = constants[STROUHAL];
    const PetscReal Cp = constants[CP];

    f0[0] -= Cp * rho * (1 + S + X[0]);
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
    ierr = IncompressibleFlow_CompleteFlowInitialization(dm, u);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode MonitorFlowAndParticleError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
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

    // compute the flow error
    for (f = 0; f < 3; ++f) {
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
    ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // get the particle data from the context
    ParticleData particlesData = (ParticleData)ctx;
    PetscInt particleCount;
    ierr = DMSwarmGetSize(particlesData->dm, &particleCount);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    // compute the average particle location
    const PetscReal *coords;
    PetscInt dims;
    PetscReal avg[3] = {0.0, 0.0, 0.0};
    ierr = DMSwarmGetField(particlesData->dm, DMSwarmPICField_coor, &dims, NULL, (void **)&coords);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    for (PetscInt p = 0; p < particleCount; p++) {
        for (PetscInt n = 0; n < dims; n++) {
            avg[n] += coords[p * dims + n] / PetscReal(particleCount);
        }
    }
    ierr = DMSwarmRestoreField(particlesData->dm, DMSwarmPICField_coor, &dims, NULL, (void **)&coords);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g] ParticleCount: %d\n",
                       (int)step,
                       (double)crtime,
                       (double)ferrors[0],
                       (double)ferrors[1],
                       (double)ferrors[2],
                       particleCount,
                       (double)avg[0],
                       (double)avg[1]);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Avg Particle Location: [%2.3g, %2.3g, %2.3g]\n", (double)avg[0], (double)avg[1], (double)avg[2]);

    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    PetscFunctionReturn(0);
}

/**
 * Computes the particle error at the specified time step.
 * @param ts
 * @param u
 * @param e
 * @return
 */
static PetscErrorCode computeParticleError(TS particleTS, Vec u, Vec e) {
    ParticleData particles;
    DM sdm;
    const PetscScalar *xp0;
    Vec exactLocationVec;
    PetscScalar *exactLocation;
    PetscReal time;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetApplicationContext(particleTS, (void **)&particles);
    CHKERRQ(ierr);
    // get the abs time for the particle evaluation, this is the ts relative time plus the time at the start of the particle ts solve
    ierr = TSGetTime(particleTS, &time);
    CHKERRQ(ierr);
    time += particles->timeInitial;

    // extract needed objects
    CHKERRQ(ierr);
    ierr = TSGetDM(particleTS, &sdm);
    CHKERRQ(ierr);

    // create a vector to hold the exact solution
    ierr = DMSwarmVectorDefineField(sdm, "InitialLocation");
    CHKERRQ(ierr);
    ierr = DMGetGlobalVector(sdm, &exactLocationVec);
    CHKERRQ(ierr);

    // use the initial location to compute the exact
    ierr = DMSwarmGetField(particles->dm, "InitialLocation", NULL, NULL, (void **)&xp0);
    CHKERRQ(ierr);
    ierr = VecGetArrayWrite(exactLocationVec, &exactLocation);
    CHKERRQ(ierr);

    // exact the exact solution from the initial location
    PetscInt dim;
    ierr = DMGetDimension(sdm, &dim);
    CHKERRQ(ierr);
    PetscInt Np;
    ierr = DMSwarmGetLocalSize(sdm, &Np);
    CHKERRQ(ierr);
    for (PetscInt p = 0; p < Np; ++p) {
        PetscScalar x[3];
        PetscReal x0[3];
        PetscInt d;

        for (d = 0; d < dim; ++d) {
            x0[d] = PetscRealPart(xp0[p * dim + d]);
        }
        ierr = particles->exactSolution(dim, time, x0, 1, x, particles->exactSolutionContext);
        CHKERRQ(ierr);
        for (d = 0; d < dim; ++d) {
            exactLocation[p * dim + d] = x[d];
        }
    }
    ierr = VecRestoreArrayWrite(exactLocationVec, &exactLocation);
    CHKERRQ(ierr);

    // Get all points still in this mesh
    DM flowDM;
    ierr = VecGetDM(particles->flowFinal, &flowDM);
    CHKERRQ(ierr);
    PetscSF cellSF = NULL;
    ierr = DMLocatePoints(flowDM, exactLocationVec, DM_POINTLOCATION_NONE, &cellSF);
    CHKERRQ(ierr);
    const PetscSFNode *cells;
    ierr = PetscSFGetGraph(cellSF, NULL, NULL, NULL, &cells);
    CHKERRQ(ierr);

    // compute the difference between exact and u
    ierr = VecWAXPY(e, -1, exactLocationVec, u);
    CHKERRQ(ierr);

    // zero out the error if any particle moves outside of the domain
    for (PetscInt p = 0; p < Np; ++p) {
        PetscInt d;
        if (cells[p].index == DMLOCATEPOINT_POINT_NOT_FOUND) {
            for (d = 0; d < dim; ++d) {
                ierr = VecSetValue(e, p * dim + d, 0.0, INSERT_VALUES);
                CHKERRQ(ierr);
            }
        }
    }
    ierr = VecAssemblyBegin(e);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(e);
    CHKERRQ(ierr);

    // restore all of the vecs/fields
    ierr = PetscSFDestroy(&cellSF);
    CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(particles->dm, "InitialLocation", NULL, NULL, (void **)&xp0);
    CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(sdm, &exactLocationVec);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/**
 * Sets the u vector to the x location at the initial time in the TS
 * @param particleTS
 * @param u
 * @return
 */
static PetscErrorCode setParticleExactSolution(TS particleTS, Vec u) {
    ParticleData particles;
    DM dm;
    PetscFunctionBegin;
    PetscErrorCode ierr = TSGetApplicationContext(particleTS, &particles);
    CHKERRQ(ierr);
    ierr = TSGetDM(particleTS, &dm);
    CHKERRQ(ierr);

    DM sdm;
    const PetscScalar *xp0;
    PetscScalar *xp;
    PetscInt dim, Np, p;
    MPI_Comm comm;

    PetscFunctionBeginUser;
    ierr = TSGetApplicationContext(particleTS, (void **)&particles);
    CHKERRQ(ierr);
    // get the abs time for the particle evaluation, this is the ts relative time plus the time at the start of the particle ts solve
    PetscReal time;
    ierr = TSGetTime(particleTS, &time);
    CHKERRQ(ierr);
    time += particles->timeInitial;

    ierr = PetscObjectGetComm((PetscObject)particleTS, &comm);
    CHKERRQ(ierr);
    ierr = TSGetDM(particleTS, &sdm);
    CHKERRQ(ierr);
    ierr = DMGetDimension(sdm, &dim);
    CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(sdm, &Np);
    CHKERRQ(ierr);
    ierr = DMSwarmGetField(particles->dm, "InitialLocation", NULL, NULL, (void **)&xp0);
    CHKERRQ(ierr);
    ierr = VecGetArrayWrite(u, &xp);
    CHKERRQ(ierr);
    for (p = 0; p < Np; ++p) {
        PetscScalar x[3];
        PetscReal x0[3];
        PetscInt d;

        for (d = 0; d < dim; ++d) x0[d] = PetscRealPart(xp0[p * dim + d]);
        ierr = particles->exactSolution(dim, time, x0, 1, x, particles->exactSolutionContext);
        CHKERRQ(ierr);
        for (d = 0; d < dim; ++d) {
            xp[p * dim + d] = x[d];
        }
    }
    ierr = DMSwarmRestoreField(particles->dm, "InitialLocation", NULL, NULL, (void **)&xp0);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(u, &xp);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

TEST_P(ParticleMMS, ParticleFlowMMSTests) {
    StartWithMPI
        DM dm;                 /* problem definition */
        TS ts;                 /* timestepper */
        PetscBag parameterBag; /* constant flow parameters */
        FlowData flowData;     /* store some of the flow data*/

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

        // setup problem
        ierr = FlowCreate(&flowData);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = IncompressibleFlow_SetupDiscretization(flowData, dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // get the flow parameters from options
        IncompressibleFlowParameters *flowParameters;
        ierr = IncompressibleFlow_ParametersFromPETScOptions(&parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscBagGetData(parameterBag, (void **)&flowParameters);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // setup problem
        PetscScalar constants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
        ierr = IncompressibleFlow_PackParameters(flowParameters, constants);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = IncompressibleFlow_StartProblemSetup(flowData, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, constants);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Override problem with source terms, boundary, and set the exact solution
        {
            PetscDS prob;
            ierr = DMGetDS(dm, &prob);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            // V, W Test Function
            IntegrandTestFunction tempFunctionPointer;
            if (testingParam.f0_v) {
                ierr = PetscDSGetResidual(prob, VTEST, &f0_v_original, &tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
                ierr = PetscDSSetResidual(prob, VTEST, testingParam.f0_v, tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
            }
            if (testingParam.f0_w) {
                ierr = PetscDSGetResidual(prob, WTEST, &f0_w_original, &tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
                ierr = PetscDSSetResidual(prob, WTEST, testingParam.f0_w, tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
            }
            if (testingParam.f0_q) {
                ierr = PetscDSGetResidual(prob, QTEST, &f0_q_original, &tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
                ierr = PetscDSSetResidual(prob, QTEST, testingParam.f0_q, tempFunctionPointer);
                CHKERRABORT(PETSC_COMM_WORLD, ierr);
            }
            /* Setup Boundary Conditions */
            PetscInt id;
            id = 3;
            ierr = PetscDSAddBoundary(
                prob, DM_BC_ESSENTIAL, "top wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 1;
            ierr = PetscDSAddBoundary(
                prob, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 2;
            ierr = PetscDSAddBoundary(
                prob, DM_BC_ESSENTIAL, "right wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 4;
            ierr = PetscDSAddBoundary(
                prob, DM_BC_ESSENTIAL, "left wall velocity", "marker", VEL, 0, NULL, (void (*)(void))testingParam.uExact, (void (*)(void))testingParam.u_tExact, 1, &id, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 3;
            ierr =
                PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 1;
            ierr = PetscDSAddBoundary(
                prob, DM_BC_ESSENTIAL, "bottom wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 2;
            ierr =
                PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 4;
            ierr =
                PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall temp", "marker", TEMP, 0, NULL, (void (*)(void))testingParam.TExact, (void (*)(void))testingParam.T_tExact, 1, &id, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            // Set the exact solution
            ierr = PetscDSSetExactSolution(prob, VEL, testingParam.uExact, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolution(prob, PRES, testingParam.pExact, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolution(prob, TEMP, testingParam.TExact, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, VEL, testingParam.u_tExact, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, PRES, NULL, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, TEMP, testingParam.T_tExact, parameterBag);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        ierr = IncompressibleFlow_CompleteProblemSetup(flowData, ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Name the flow field
        ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);
        CHKERRABORT(PETSC_COMM_WORLD, ierr); /* Must come after SetFromOptions() */
        ierr = SetInitialConditions(ts, flowData->flowField);

        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSGetTime(ts, &t);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSetOutputSequenceNumber(dm, 0, t);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMTSCheckFromOptions(ts, flowData->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ParticleData particles;

        // Setup the particle domain
        ierr = ParticleTracerCreate(&particles, 2);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // For this test suite keep track of the initial particle location
        ierr = ParticleRegisterPetscDatatypeField(particles, "InitialLocation", 2, PETSC_REAL);

        // Set the exact solution
        ParticleSetExactSolutionFlow(particles, testingParam.particleExact, NULL);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // link the flow to the particles
        ParticleInitializeFlow(particles, flowData);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // name the particle domain
        ierr = PetscObjectSetOptionsPrefix((PetscObject)(particles->dm), "particles_");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscObjectSetName((PetscObject)particles->dm, "Particles");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // initialize the particles
        ParticleInitialize(dm, particles->dm);

        // setup the flow monitor to also check particles
        ierr = TSMonitorSet(ts, MonitorFlowAndParticleError, particles, NULL);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSSetFromOptions(ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Setup particle position integrator
        TS particleTs;
        ierr = TSCreate(PETSC_COMM_WORLD, &particleTs);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject)particleTs, "particle_");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = ParticleTracerSetupIntegrator(particles, particleTs, flowData);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // setup the initial conditions for error computing
        ierr = TSSetComputeExactError(particleTs, computeParticleError);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSSetComputeInitialCondition(particleTs, setParticleExactSolution);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // copy over the initial location
        PetscReal *coord;
        PetscReal *initialLocation;
        PetscInt numberParticles;
        ierr = DMSwarmGetLocalSize(particles->dm, &numberParticles);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSwarmGetField(particles->dm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSwarmGetField(particles->dm, "InitialLocation", NULL, NULL, (void **)&initialLocation);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        for (int i = 0; i < numberParticles * 2; ++i) {
            initialLocation[i] = coord[i];
        }
        ierr = DMSwarmRestoreField(particles->dm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSwarmRestoreField(particles->dm, "InitialLocation", NULL, NULL, (void **)&initialLocation);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Solve the one way coupled system
        ierr = TSSolve(ts, flowData->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Compare the actual vs expected values
        ierr = DMTSCheckFromOptions(ts, flowData->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Cleanup
        ierr = DMDestroy(&dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSDestroy(&ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSDestroy(&particleTs);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = FlowDestroy(&flowData);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = ParticleTracerDestroy(&particles);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscFinalize();
        exit(ierr);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(ParticleMMSTests, ParticleMMS,
                         testing::Values((ParticleMMSParameters){.mpiTestParameter = {.testName = "particle in incompressible 2d trigonometric trigonometric tri_p2_p1_p1",
                                                                                      .nproc = 1,
                                                                                      .expectedOutputFile = "outputs/particle_incompressible_trigonometric_2d_tri_p2_p1_p1",
                                                                                      .arguments = "-dm_plex_separate_marker -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 "
                                                                                                   "-temp_petscspace_degree 1 -dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 -ts_monitor_cancel "
                                                                                                   "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                                                                                   "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 "
                                                                                                   "-pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full -fieldsplit_0_pc_type lu "
                                                                                                   "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi -particle_layout_type box "
                                                                                                   "-particle_lower 0.25,0.25 -particle_upper 0.75,0.75 -Npb 5 "
                                                                                                   "-particle_ts_dt 0.05 -particle_ts_convergence_estimate -convest_num_refine 1 "
                                                                                                   "-particle_ts_monitor_cancel"},
                                                                 .uExact = trig_trig_u,
                                                                 .pExact = trig_trig_p,
                                                                 .TExact = trig_trig_T,
                                                                 .u_tExact = trig_trig_u_t,
                                                                 .T_tExact = trig_trig_T_t,
                                                                 .particleExact = trig_trig_x,
                                                                 .f0_v = f0_trig_trig_v,
                                                                 .f0_w = f0_trig_trig_w,
                                                                 .omega = 0.5},
                                         (ParticleMMSParameters){.mpiTestParameter = {.testName = "particle deletion with simple fluid tri_p2_p1_p1",
                                                                                      .nproc = 1,
                                                                                      .expectedOutputFile = "outputs/particle_deletion_with_simple_fluid_tri_p2_p1_p1",
                                                                                      .arguments = "-dm_plex_separate_marker -dm_refine 2 "
                                                                                                   "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                                                                                   "-dmts_check .001 -ts_max_steps 7 -ts_dt 0.06 -ksp_type fgmres -ksp_gmres_restart 10 "
                                                                                                   "-ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 "
                                                                                                   "-pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                                                                   "-fieldsplit_0_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                                                                                   "-particle_layout_type box -particle_lower 0.25,0.25 -particle_upper 0.75,0.75 -Npb 5 "
                                                                                                   "-particle_ts_dt 0.03 -particle_ts_convergence_estimate -convest_num_refine 1 "},
                                                                 .uExact = linear_u,
                                                                 .pExact = linear_p,
                                                                 .TExact = linear_T,
                                                                 .u_tExact = linear_u_t,
                                                                 .T_tExact = linear_T_t,
                                                                 .particleExact = linear_x,
                                                                 .f0_v = f0_linear_v,
                                                                 .f0_w = f0_linear_w}),
                         [](const testing::TestParamInfo<ParticleMMSParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
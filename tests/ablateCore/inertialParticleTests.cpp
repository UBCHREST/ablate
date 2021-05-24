#include <inttypes.h>
#include <petsc.h>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"
#include "incompressibleFlow.h"
#include "mesh.h"
#include "particleInertial.h"
#include "particleInitializer.h"
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
static PetscInt totalFields = 2;
static PetscInt dim;
static PetscReal pVel;
static PetscReal dp;
static PetscReal rhoP;
static PetscReal rhoF;
static PetscReal muF;
static PetscReal grav;

struct InertialParticleMMSParameters {
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
    PetscInt dim;
    PetscReal pVel;  // particle initial velocity
    PetscReal dp;
    PetscReal rhoP;
    PetscReal rhoF;
    PetscReal muF;
    PetscReal grav;
};

class InertialParticleMMS : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<InertialParticleMMSParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscErrorCode settling(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *x, void *ctx) {
    const PetscReal x0 = X[0];
    const PetscReal y0 = X[1];
    PetscReal tauP = rhoP * dp * dp / (18.0 * muF);     // particle relaxation time
    PetscReal uSt = tauP * grav * (1.0 - rhoF / rhoP);  // particle terminal (settling) velocity
    x[0] = uSt * (time + tauP * PetscExpReal(-time / tauP) - tauP) + x0;
    x[1] = y0;
    x[2] = uSt * (1.0 - PetscExpReal(-time / tauP));
    x[3] = 0.0;
    return 0;
}

static PetscErrorCode quiescent_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Dim; ++d) u[d] = 0.0;
    return 0;
}

static PetscErrorCode quiescent_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscInt d;
    for (d = 0; d < Dim; ++d) u[d] = 0.0;
    return 0;
}

static PetscErrorCode quiescent_p(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx) {
    p[0] = 0.0;
    return 0;
}

static PetscErrorCode quiescent_T(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 0.0;
    return 0;
}
static PetscErrorCode quiescent_T_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *T, void *ctx) {
    T[0] = 0.0;
    return 0;
}

static void SourceFunction(f0_quiescent_v) {
    f0_v_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] = 0;
    f0[1] = 0;
}

static void SourceFunction(f0_quiescent_w) {
    f0_w_original(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, a_t, a_x, t, X, numConstants, constants, f0);
    f0[0] = 0;
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
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);  // exsatFuncs are output
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
    for (PetscInt n = 0; n < dims; n++) {
        for (PetscInt p = 0; p < particleCount; p++) {
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
                       particleCount);
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
    Vec exactSolutionVec, exactPositionVec;
    PetscScalar *exactSolution, *exactPosition;
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
    ierr = DMSwarmVectorDefineField(sdm, "InitialSolution");
    CHKERRQ(ierr);
    ierr = DMGetGlobalVector(sdm, &exactSolutionVec);
    CHKERRQ(ierr);

    // create a vector to hold the exact position
    ierr = DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor);
    CHKERRQ(ierr);
    ierr = DMGetGlobalVector(sdm, &exactPositionVec);
    CHKERRQ(ierr);

    // use the initial solution to compute the exact
    ierr = DMSwarmGetField(particles->dm, "InitialSolution", NULL, NULL, (void **)&xp0);
    CHKERRQ(ierr);

    // get two vectors for position and solution
    ierr = VecGetArrayWrite(exactSolutionVec, &exactSolution);
    CHKERRQ(ierr);
    ierr = VecGetArrayWrite(exactPositionVec, &exactPosition);
    CHKERRQ(ierr);

    // exact the exact solution from the initial location
    PetscInt dim;
    ierr = DMGetDimension(sdm, &dim);
    CHKERRQ(ierr);
    PetscInt Np;
    ierr = DMSwarmGetLocalSize(sdm, &Np);
    CHKERRQ(ierr);
    for (PetscInt p = 0; p < Np; ++p) {
        PetscScalar x[4];                        // includes both position and velocity in 2D
        PetscReal x0[4] = {0.0, 0.0, 0.0, 0.0};  // includes both initial position and velocity in 2D
        PetscInt d;

        for (d = 0; d < totalFields * dim; ++d) {
            x0[d] = PetscRealPart(xp0[p * totalFields * dim + d]);
        }
        ierr = particles->exactSolution(dim, time, x0, 1, x, particles->exactSolutionContext);
        CHKERRQ(ierr);
        for (d = 0; d < totalFields * dim; ++d) {
            exactSolution[p * totalFields * dim + d] = x[d];  // exactSolution including velocity and position
            if (d < 2) {
                exactPosition[p * dim + d] = x[d];  // stores the position only and hard coded for 2D cases
            }
        }
    }
    ierr = VecRestoreArrayWrite(exactSolutionVec, &exactSolution);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(exactPositionVec, &exactPosition);
    CHKERRQ(ierr);

    // compute the difference between exact and u
    ierr = VecWAXPY(e, -1, exactSolutionVec, u);
    CHKERRQ(ierr);

    // Get all points still in this mesh
    DM flowDM;
    ierr = VecGetDM(particles->flowFinal, &flowDM);
    CHKERRQ(ierr);
    PetscSF cellSF = NULL;
    ierr = DMLocatePoints(flowDM, exactPositionVec, DM_POINTLOCATION_NONE, &cellSF);
    CHKERRQ(ierr);
    const PetscSFNode *cells;
    ierr = PetscSFGetGraph(cellSF, NULL, NULL, NULL, &cells);
    CHKERRQ(ierr);

    // zero out the error if any particle moves outside of the domain
    for (PetscInt p = 0; p < Np; ++p) {
        PetscInt d;
        if (cells[p].index == DMLOCATEPOINT_POINT_NOT_FOUND) {
            for (d = 0; d < totalFields * dim; ++d) {
                ierr = VecSetValue(e, p * totalFields * dim + d, 0.0, INSERT_VALUES);
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
    ierr = DMSwarmRestoreField(particles->dm, "InitialSolution", NULL, NULL, (void **)&xp0);
    CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(sdm, &exactSolutionVec);
    CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(sdm, &exactPositionVec);
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
    ierr = DMSwarmGetField(particles->dm, "InitialSolution", NULL, NULL, (void **)&xp0);
    CHKERRQ(ierr);
    ierr = VecGetArrayWrite(u, &xp);
    CHKERRQ(ierr);
    for (p = 0; p < Np; ++p) {
        PetscScalar x[4];  // includes position and velocity in 2D
        PetscReal x0[4];   // includes initial position and velocity in 2D
        PetscInt d;

        for (d = 0; d < totalFields * dim; ++d) x0[d] = PetscRealPart(xp0[p * totalFields * dim + d]);
        ierr = particles->exactSolution(dim, time, x0, 1, x, particles->exactSolutionContext);
        CHKERRQ(ierr);
        for (d = 0; d < totalFields * dim; ++d) {
            xp[p * totalFields * dim + d] = x[d];
        }
    }
    ierr = DMSwarmRestoreField(particles->dm, "InitialSolution", NULL, NULL, (void **)&xp0);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(u, &xp);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode ParticleInertialInitialize(ParticleData particles, PetscReal partVel, PetscReal partDiam, PetscReal partDens, PetscReal fluidDens, PetscReal fluidVisc, PetscReal gravity) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    Vec vel, diam, dens;
    DM particleDm = particles->dm;

    ierr = DMSwarmCreateGlobalVectorFromField(particleDm, ParticleVelocity, &vel);
    CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(particleDm, ParticleDiameter, &diam);
    CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(particleDm, ParticleDensity, &dens);
    CHKERRQ(ierr);
    // set particle velocity, diameter and density
    ierr = VecSet(vel, partVel);
    CHKERRQ(ierr);
    ierr = VecSet(diam, partDiam);
    CHKERRQ(ierr);
    ierr = VecSet(dens, partDens);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(particleDm, ParticleVelocity, &vel);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(particleDm, ParticleDiameter, &diam);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(particleDm, ParticleDensity, &dens);
    CHKERRQ(ierr);

    InertialParticleParameters *data;
    PetscNew(&data);
    particles->data = data;

    // set fluid parameters
    data->fluidDensity = fluidDens;
    data->fluidViscosity = fluidVisc;
    data->gravityField[0] = gravity;  // only for one direction
    data->gravityField[1] = 0.0;      // zero out other components
    data->gravityField[2] = 0.0;
    PetscFunctionReturn(0);
}

TEST_P(InertialParticleMMS, ParticleFlowMMSTests) {
    StartWithMPI
        DM dm;                 /* problem definition */
        TS ts;                 /* timestepper */
        PetscBag parameterBag; /* constant flow parameters */
        FlowData flowData;     /* store some of the flow data*/

        PetscReal t;
        PetscErrorCode ierr;

        // Get the testing param
        auto testingParam = GetParam();
        pVel = testingParam.pVel;  // particle initial velocity
        dp = testingParam.dp;
        rhoP = testingParam.rhoP;
        rhoF = testingParam.rhoF;
        muF = testingParam.muF;
        grav = testingParam.grav;
        PetscInt dimen = testingParam.dim;

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
        ierr = ParticleInertialCreate(&particles, dimen);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // For this test suite keep track of the initial particle location
        ierr = ParticleRegisterPetscDatatypeField(particles, "InitialSolution", totalFields * dimen, PETSC_REAL);

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

        // initialize the particles position
        ierr = ParticleInitialize(dm, particles->dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // initialize inertial particles with velocity, diameter and density
        ierr = ParticleInertialInitialize(particles, pVel, dp, rhoP, rhoF, muF, grav);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

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

        ierr = ParticleInertialSetupIntegrator(particles, particleTs, flowData);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // setup the initial conditions for error computing
        ierr = TSSetComputeExactError(particleTs, computeParticleError);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSSetComputeInitialCondition(particleTs, setParticleExactSolution);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // copy over the initial location
        PetscReal *coord, *partVel;
        PetscReal *initialSolution;
        PetscInt numberParticles;
        ierr = DMSwarmGetLocalSize(particles->dm, &numberParticles);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSwarmGetField(particles->dm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSwarmGetField(particles->dm, ParticleVelocity, NULL, NULL, (void **)&partVel);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = DMSwarmGetField(particles->dm, "InitialSolution", NULL, NULL, (void **)&initialSolution);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // filling initialSolution with particles initial position and velocity
        PetscInt d, p;
        for (p = 0; p < numberParticles; ++p) {
            for (d = 0; d < dimen; d++) {
                initialSolution[p * totalFields * dimen + d] = coord[p * dimen + d];
                initialSolution[p * totalFields * dimen + dimen + d] = partVel[p * dimen + d];
            }
        }

        ierr = DMSwarmRestoreField(particles->dm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSwarmRestoreField(particles->dm, ParticleVelocity, NULL, NULL, (void **)&partVel);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMSwarmRestoreField(particles->dm, "InitialSolution", NULL, NULL, (void **)&initialSolution);
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
        ierr = ParticleInertialDestroy(&particles);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscFinalize();
        exit(ierr);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(InertialParticleMMSTests, InertialParticleMMS,
                         testing::Values((InertialParticleMMSParameters){.mpiTestParameter = {.testName = "single inertial particle settling in quiescent fluid",
                                                                                              .nproc = 1,
                                                                                              .expectedOutputFile = "outputs/single_inertialParticle_settling_in_quiescent_fluid",
                                                                                              .arguments = "-dm_plex_separate_marker -dm_refine 2 "
                                                                                                           "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                                                                                           "-dmts_check .001 -ts_max_steps 7 -ts_dt 0.06 -ksp_type fgmres -ksp_gmres_restart 10 "
                                                                                                           "-ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_type fieldsplit  "
                                                                                                           " -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                                                                           "-particle_layout_type box -particle_lower 0.5,0.5 -particle_upper 0.5,0.5 -Npb 1 "
                                                                                                           "-particle_ts_dt 0.03 -particle_ts_convergence_estimate -convest_num_refine 1 "},
                                                                         .uExact = quiescent_u,
                                                                         .pExact = quiescent_p,
                                                                         .TExact = quiescent_T,
                                                                         .u_tExact = quiescent_u_t,
                                                                         .T_tExact = quiescent_T_t,
                                                                         .particleExact = settling,
                                                                         .f0_v = f0_quiescent_v,
                                                                         .f0_w = f0_quiescent_w,
                                                                         .dim = 2,
                                                                         .pVel = 0.0,
                                                                         .dp = 0.22,
                                                                         .rhoP = 90.0,
                                                                         .rhoF = 1.0,
                                                                         .muF = 1.0,
                                                                         .grav = 1.0},
                                         (InertialParticleMMSParameters){.mpiTestParameter = {.testName = "multi inertial particle settling in quiescent fluid",
                                                                                              .nproc = 1,
                                                                                              .expectedOutputFile = "outputs/multi_inertialParticle_settling_in_quiescent_fluid",
                                                                                              .arguments = "-dm_plex_separate_marker -dm_refine 2 "
                                                                                                           "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                                                                                           "-dmts_check .001 -ts_max_steps 7 -ts_dt 0.06 -ksp_type fgmres -ksp_gmres_restart 10 "
                                                                                                           "-ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_type fieldsplit  "
                                                                                                           " -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                                                                           "-particle_layout_type box -particle_lower 0.2,0.3 -particle_upper 0.4,0.6 -Npb 10 "
                                                                                                           "-particle_ts_dt 0.03 -particle_ts_convergence_estimate -convest_num_refine 1 "},
                                                                         // --keepOutputFile=true --inmpitestrun=true
                                                                         .uExact = quiescent_u,
                                                                         .pExact = quiescent_p,
                                                                         .TExact = quiescent_T,
                                                                         .u_tExact = quiescent_u_t,
                                                                         .T_tExact = quiescent_T_t,
                                                                         .particleExact = settling,
                                                                         .f0_v = f0_quiescent_v,
                                                                         .f0_w = f0_quiescent_w,
                                                                         .dim = 2,
                                                                         .pVel = 0.0,
                                                                         .dp = 0.22,
                                                                         .rhoP = 90.0,
                                                                         .rhoF = 1.0,
                                                                         .muF = 1.0,
                                                                         .grav = 1.0},
                                         (InertialParticleMMSParameters){.mpiTestParameter = {.testName = "deletion inertial particles settling in quiescent fluid",
                                                                                              .nproc = 1,
                                                                                              .expectedOutputFile = "outputs/deletion_inertialParticles_settling_in_quiescent_fluid",
                                                                                              .arguments = "-dm_plex_separate_marker -dm_refine 2 "
                                                                                                           "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                                                                                           "-dmts_check .001 -ts_max_steps 7 -ts_dt 0.06 -ksp_type fgmres -ksp_gmres_restart 10 "
                                                                                                           "-ksp_rtol 1.0e-9 -ksp_error_if_not_converged -pc_type fieldsplit  "
                                                                                                           " -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                                                                           "-particle_layout_type box -particle_lower 0.92,0.3 -particle_upper 0.98,0.6 -Npb 10 "
                                                                                                           "-particle_ts_dt 0.03 -particle_ts_convergence_estimate -convest_num_refine 1 "},
                                                                         // --keepOutputFile=true --inmpitestrun=true
                                                                         .uExact = quiescent_u,
                                                                         .pExact = quiescent_p,
                                                                         .TExact = quiescent_T,
                                                                         .u_tExact = quiescent_u_t,
                                                                         .T_tExact = quiescent_T_t,
                                                                         .particleExact = settling,
                                                                         .f0_v = f0_quiescent_v,
                                                                         .f0_w = f0_quiescent_w,
                                                                         .dim = 2,
                                                                         .pVel = 0.0,
                                                                         .dp = 0.22,
                                                                         .rhoP = 90.0,
                                                                         .rhoF = 1.0,
                                                                         .muF = 1.0,
                                                                         .grav = 1.0}),
                         [](const testing::TestParamInfo<InertialParticleMMSParameters> &info) { return info.param.mpiTestParameter.getTestName(); });

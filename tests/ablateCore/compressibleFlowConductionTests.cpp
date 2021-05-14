static char help[] = "1D conduction and diffusion cases compared to exact solution";

#include <compressibleFlow.h>
#include <petsc.h>
#include <vector>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"

typedef struct {
    PetscInt dim;
    PetscReal L;
    PetscReal gamma;
    PetscReal Rgas;
    PetscReal k;
    PetscReal rho;
    PetscReal Tinit;
} InputParameters;

struct CompressibleFlowDiffusionTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    InputParameters parameters;
    PetscInt initialNx;
    std::vector<PetscReal> expectedL2Convergence;
    std::vector<PetscReal> expectedLInfConvergence;
};

//typedef struct {
//    Constants constants;
//    FlowData flowData;
//} ProblemSetup;

class CompressibleFlowDiffusionTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<CompressibleFlowDiffusionTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

/**
 * Simple function to compute the exact solution for a given xyz and time
 */
static PetscReal ComputeTExact( PetscReal time, const PetscReal xyz[], InputParameters *parameters) {
    // compute cv for a perfect gas
    PetscReal cv = parameters->gamma*parameters->Rgas/(parameters->gamma - 1) - parameters->Rgas;

    // compute the alpha in the equation
    PetscReal alpha = parameters->k/(parameters->rho*cv);
    PetscReal Tinitial = parameters->Tinit;
    PetscReal T = 0.0;
    for(PetscReal n =1; n < 2000; n ++){
        PetscReal Bn = -Tinitial*2.0*(-1.0 + PetscPowReal(-1.0, n))/(n*PETSC_PI);
        T += Bn*PetscSinReal(n * PETSC_PI*xyz[0]/parameters->L)*PetscExpReal(-n*n*PETSC_PI*PETSC_PI*alpha*time/(PetscSqr(parameters->L)));
    }

    return T;
}

static PetscErrorCode EulerExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar *node, void *ctx) {
    PetscFunctionBeginUser;

    InputParameters *parameters = (InputParameters *)ctx;

    PetscReal T = ComputeTExact(time, xyz, parameters);

    PetscReal u = 0.0;
    PetscReal v = 0.0;
    PetscReal p= parameters->rho*parameters->Rgas*T;
    PetscReal e = p/((parameters->gamma - 1.0)*parameters->rho);
    PetscReal eT = e + 0.5*(u*u + v*v);

    node[RHO] = parameters->rho;
    node[RHOE] = parameters->rho*eT;
    node[RHOU + 0] = parameters->rho*u;
    node[RHOU + 1] = parameters->rho*v;

    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    InputParameters *parameters = (InputParameters *)ctx;

    // compute the centroid location of the real cell
    // Offset the calc assuming the cells are square
    PetscReal x[3];
    for(PetscInt i =0; i < parameters->dim; i++){
        x[i] = c[i] - n[i]*0.5;
    }

    // compute the temperature of the real inside cell
    PetscReal Tinside =   ComputeTExact(time, x, parameters);
    PetscReal boundaryValue = 0.0;

    PetscReal T = boundaryValue - (Tinside - boundaryValue);
    PetscReal u = 0.0;
    PetscReal v = 0.0;
    PetscReal p= parameters->rho*parameters->Rgas*T;
    PetscReal e = p/((parameters->gamma - 1.0)*parameters->rho);
    PetscReal eT = e + 0.5*(u*u + v*v);

    a_xG[RHO] = parameters->rho;
    a_xG[RHOE] = parameters->rho*eT;
    a_xG[RHOU + 0] = parameters->rho*u;
    a_xG[RHOU + 1] = parameters->rho*v;

    PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsBoundary_Mirror(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    PetscFunctionBeginUser;
    InputParameters *constants = (InputParameters *)ctx;

    // Offset the calc assuming the cells are square
    for(PetscInt f =0; f < RHOU + constants->dim; f++){
        a_xG[f] = a_xI[f];
    }
    PetscFunctionReturn(0);
}

TEST_P(CompressibleFlowDiffusionTestFixture, ShouldConvergeToExactSolution) {
    StartWithMPI
        PetscErrorCode ierr;

        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, "HELP") >> errorChecker;

        PetscInt levels = 4;

        InputParameters parameters = GetParam().parameters;
        parameters.dim = 2;
        PetscInt blockSize = 2 + parameters.dim;
        PetscInt initialNx = GetParam().initialNx;

        std::vector<PetscReal> hHistory;
        std::vector<std::vector<PetscReal>> l2History(blockSize);
        std::vector<std::vector<PetscReal>> lInfHistory(blockSize);

        // March over each level
        for (PetscInt l = 0; l < levels; l++) {
            PetscPrintf(PETSC_COMM_WORLD, "Running RHS Calculation at Level %d", l);

            DM dm; /* problem definition */
            TS ts; /* timestepper */

            // Create a ts
            TSCreate(PETSC_COMM_WORLD, &ts) >> errorChecker;
            TSSetProblemType(ts, TS_NONLINEAR) >> errorChecker;
            TSSetType(ts, TSEULER) >> errorChecker;
            TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> errorChecker;
            TSSetFromOptions(ts) >> errorChecker;


            // Create a mesh
            // hard code the problem setup
            PetscReal start[] = {0.0, 0.0};
            PetscReal end[] = {parameters.L, parameters.L};
            PetscInt nx1D = initialNx * PetscPowRealInt(2, l);
            PetscInt nx[] = {nx1D, nx1D};
            DMBoundaryType bcType[] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
            DMPlexCreateBoxMesh(PETSC_COMM_WORLD, parameters.dim, PETSC_FALSE, nx, start, end, bcType, PETSC_TRUE, &dm) >> errorChecker;

            // Setup the flow data
            FlowData flowData; /* store some of the flow data*/
            FlowCreate(&flowData) >> errorChecker;

            // Setup
            CompressibleFlow_SetupDiscretization(flowData, &dm);

            // Add in the flow parameters
            PetscScalar params[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS];
            params[CFL] = 0.5;
            params[GAMMA] = parameters.gamma;
            params[RGAS] = parameters.Rgas;
            params[K] = parameters.k;

            // set up the finite volume fluxes
            CompressibleFlow_StartProblemSetup(flowData, TOTAL_COMPRESSIBLE_FLOW_PARAMETERS, params) >> errorChecker;

            // Add in any boundary conditions
            PetscDS prob;
            ierr = DMGetDS(flowData->dm, &prob);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            const PetscInt idsLeft[]= {2, 4};
            ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "wall left", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Euler, NULL, 2, idsLeft, &constants);CHKERRQ(ierr);

            const PetscInt idsTop[]= {1, 3};
            ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL_RIEMANN, "top/bottom", "Face Sets", 0, 0, NULL, (void (*)(void))PhysicsBoundary_Mirror, NULL, 2, idsTop, &constants);CHKERRQ(ierr);

            // Complete the problem setup
            CompressibleFlow_CompleteProblemSetup(flowData, ts) >> errorChecker;

            // Name the flow field
            PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution") >> errorChecker;

            // Setup the TS
            TSSetFromOptions(ts) >> errorChecker;

            // set the initial conditions
            PetscErrorCode     (*func[2]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) = {EulerExact};
            void* ctxs[1] ={&parameters};
            DMProjectFunction(flowData->dm,0.0,func,ctxs,INSERT_ALL_VALUES,flowData->flowField) >> errorChecker;

            // for the mms, add the exact solution
            PetscDSSetExactSolution(prob, 0, EulerExact, &parameters) >> errorChecker;

            TSSolve(ts, flowData->flowField) >> errorChecker;

            // Check the current residual
            PetscReal l2Residual[5];
            PetscReal infResidual[5];

            ComputeRHS(ts, flowData->dm, 0.0, flowData->flowField, blockSize, l2Residual, infResidual, resStart, resEnd) >> errorChecker;
            PetscPrintf(PETSC_COMM_WORLD, "\tL_2 Residual: [%2.3g, %2.3g, %2.3g, %2.3g]\n", (double)l2Residual[0], (double)l2Residual[1], (double)l2Residual[2], (double)l2Residual[3]) >> errorChecker;
            PetscPrintf(PETSC_COMM_WORLD, "\tL_Inf Residual: [%2.3g, %2.3g, %2.3g, %2.3g]\n", (double)infResidual[0], (double)infResidual[1], (double)infResidual[2], (double)infResidual[3]) >>
                errorChecker;

            // Store the residual into history
            hHistory.push_back(PetscLog10Real(constants.L / nx1D));
            for (auto b = 0; b < blockSize; b++) {
                l2History[b].push_back(PetscLog10Real(l2Residual[b]));
                lInfHistory[b].push_back(PetscLog10Real(infResidual[b]));
            }

            FlowDestroy(&flowData) >> errorChecker;
            TSDestroy(&ts) >> errorChecker;
        }

        // Fit each component and output
        for (auto b = 0; b < blockSize; b++) {
            PetscReal l2Slope;
            PetscReal l2Intercept;
            PetscLinearRegression(hHistory.size(), &hHistory[0], &l2History[b][0], &l2Slope, &l2Intercept) >> errorChecker;

            PetscReal lInfSlope;
            PetscReal lInfIntercept;
            PetscLinearRegression(hHistory.size(), &hHistory[0], &lInfHistory[b][0], &lInfSlope, &lInfIntercept) >> errorChecker;

            PetscPrintf(PETSC_COMM_WORLD, "RHS Convergence[%d]: L2 %2.3g LInf %2.3g \n", b, l2Slope, lInfSlope) >> errorChecker;

            ASSERT_NEAR(l2Slope, GetParam().expectedL2Convergence[b], 0.2) << "incorrect L2 convergence order for component[" << b << "]";
            ASSERT_NEAR(lInfSlope, GetParam().expectedLInfConvergence[b], 0.2) << "incorrect LInf convergence order for component[" << b << "]";
        }

        ierr = PetscFinalize();
        exit(ierr);

    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    CompressibleFlow, CompressibleFlowMmsTestFixture,
    testing::Values((CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed average", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff average"},
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4},
                                                        .initialNx = 4,
                                                        .expectedL2Convergence = {2, 2, 2, 2},
                                                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "high speed average", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff average"},
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4},
                                                        .initialNx = 4,
                                                        .expectedL2Convergence = {2, 2, 2, 2},
                                                        .expectedLInfConvergence = {1.9, 1.8, 1.8, 1.8}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "low speed ausm", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff ausm"},
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4},
                                                        .initialNx = 16,
                                                        .expectedL2Convergence = {1.0, 1.0, 1.4, 1.0},
                                                        .expectedLInfConvergence = {1.0, 1.0, 1.4, 1.0}},
                    (CompressibleFlowMmsTestParameters){.mpiTestParameter = {.testName = "high speed ausm", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff ausm"},
                                                        .constants = {.dim = 2,
                                                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                                                      .L = 1.0,
                                                                      .gamma = 1.4},
                                                        .initialNx = 16,
                                                        .expectedL2Convergence = {1.0, 1.0, 1.0, 1.0},
                                                        .expectedLInfConvergence = {1.0, 1.0, 1.0, 1.0}},
                    (CompressibleFlowMmsTestParameters){
                        .mpiTestParameter = {.testName = "low speed ausm leastsquares", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff ausm -eulerpetscfv_type leastsquares"},
                        .constants = {.dim = 2,
                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                      .u = {.phiO = 70, .phiX = 5, .phiY = -7, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                      .v = {.phiO = 90, .phiX = -15, .phiY = -8.5, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                      .L = 1.0,
                                      .gamma = 1.4},
                        .initialNx = 16,
                        .expectedL2Convergence = {1.5, 1.5, 1.5, 1.5},
                        .expectedLInfConvergence = {1.0, 1.0, 1.0, 1.0}},
                    (CompressibleFlowMmsTestParameters){
                        .mpiTestParameter = {.testName = "high speed ausm leastsquares", .nproc = 1, .arguments = "-dm_plex_separate_marker -flux_diff ausm -eulerpetscfv_type leastsquares"},
                        .constants = {.dim = 2,
                                      .rho = {.phiO = 1.0, .phiX = 0.15, .phiY = -0.1, .phiZ = 0.0, .aPhiX = 1.0, .aPhiY = 0.5, .aPhiZ = 0.0},
                                      .u = {.phiO = 800, .phiX = 50, .phiY = -30.0, .phiZ = 0., .aPhiX = 1.5, .aPhiY = 0.6, .aPhiZ = 0.0},
                                      .v = {.phiO = 800, .phiX = -75, .phiY = 40, .phiZ = 0.0, .aPhiX = 0.5, .aPhiY = 2.0 / 3.0, .aPhiZ = 0.0},
                                      .w = {.phiO = 0.0, .phiX = 0.0, .phiY = 0.0, .phiZ = 0.0, .aPhiX = 0.0, .aPhiY = 0.0, .aPhiZ = 0.0},
                                      .p = {.phiO = 1E5, .phiX = 0.2E5, .phiY = 0.5E5, .phiZ = 0.0, .aPhiX = 2.0, .aPhiY = 1.0, .aPhiZ = 0.0},
                                      .L = 1.0,
                                      .gamma = 1.4},
                        .initialNx = 16,
                        .expectedL2Convergence = {1.5, 1.5, 1.5, 1.5},
                        .expectedLInfConvergence = {1.0, 0.5, 1.0, 1.0}}),
    [](const testing::TestParamInfo<CompressibleFlowMmsTestParameters> &info) { return info.param.mpiTestParameter.getTestName(); });

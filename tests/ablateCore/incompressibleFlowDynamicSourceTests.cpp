static char help[] =
    "Time-dependent Low Mach Flow in 2d channels with finite elements.\n\
We solve the incompressible problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include <petsc.h>
#include "MpiTestFixture.hpp"
#include "flow.h"
#include "gtest/gtest.h"
#include "incompressibleFlow.h"
#include "mesh.h"
#include "support/testingAuxFieldUpdater.hpp"
using namespace tests::ablateCore::support;

struct IncompressibleFlowDynamicSourceMMSParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::string uExact;
    std::string pExact;
    std::string TExact;
    std::string vSource;
    std::string wSource;
    std::string qSource;
};

class IncompressibleFlowDynamicSourceMMS : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<IncompressibleFlowDynamicSourceMMSParameters> {
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

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void *ctxs[3];
    DM dm;
    PetscDS ds;
    PetscReal ferrors[3];
    PetscInt f;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    ierr = VecViewFromOptions(u, NULL, "-vec_view_monitor");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    for (f = 0; f < 3; ++f) {
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
    ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int)step, (double)crtime, (double)ferrors[0], (double)ferrors[1], (double)ferrors[2]);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    PetscFunctionReturn(0);
}

TEST_P(IncompressibleFlowDynamicSourceMMS, ShouldConvergeToExactSolution) {
    StartWithMPI
        DM dm;                 /* problem definition */
        TS ts;                 /* timestepper */
        PetscBag parameterBag; /* constant flow parameters */
        FlowData flowData;     /* store some of the flow data*/
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

        // Setup the flow data
        ierr = FlowCreate(&flowData);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // setup problem
        ierr = IncompressibleFlow_SetupDiscretization(flowData, dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // get the flow parameters from options
        IncompressibleFlowParameters *flowParameters;
        ierr = IncompressibleFlow_ParametersFromPETScOptions(&parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscBagGetData(parameterBag, (void **)&flowParameters);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Start the problem setup
        PetscScalar constants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];
        ierr = IncompressibleFlow_PackParameters(flowParameters, constants);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = IncompressibleFlow_StartProblemSetup(flowData, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS, constants);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        PetscTestingFunction uExact(testingParam.uExact);
        PetscTestingFunction pExact(testingParam.pExact);
        PetscTestingFunction TExact(testingParam.TExact);

        // Override problem with source terms, boundary, and set the exact solution
        {
            PetscDS prob;
            ierr = DMGetDS(dm, &prob);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            /* Setup Boundary Conditions */
            PetscInt id;
            id = 3;
            ierr = PetscDSAddBoundary(prob,
                                      DM_BC_ESSENTIAL,
                                      "top wall velocity",
                                      "marker",
                                      VEL,
                                      0,
                                      NULL,
                                      (void (*)(void))PetscTestingFunction::ApplySolution,
                                      (void (*)(void))PetscTestingFunction::ApplySolutionTimeDerivative,
                                      1,
                                      &id,
                                      &uExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 1;
            ierr = PetscDSAddBoundary(prob,
                                      DM_BC_ESSENTIAL,
                                      "bottom wall velocity",
                                      "marker",
                                      VEL,
                                      0,
                                      NULL,
                                      (void (*)(void))PetscTestingFunction::ApplySolution,
                                      (void (*)(void))PetscTestingFunction::ApplySolutionTimeDerivative,
                                      1,
                                      &id,
                                      &uExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 2;
            ierr = PetscDSAddBoundary(prob,
                                      DM_BC_ESSENTIAL,
                                      "right wall velocity",
                                      "marker",
                                      VEL,
                                      0,
                                      NULL,
                                      (void (*)(void))PetscTestingFunction::ApplySolution,
                                      (void (*)(void))PetscTestingFunction::ApplySolutionTimeDerivative,
                                      1,
                                      &id,
                                      &uExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 4;
            ierr = PetscDSAddBoundary(prob,
                                      DM_BC_ESSENTIAL,
                                      "left wall velocity",
                                      "marker",
                                      VEL,
                                      0,
                                      NULL,
                                      (void (*)(void))PetscTestingFunction::ApplySolution,
                                      (void (*)(void))PetscTestingFunction::ApplySolutionTimeDerivative,
                                      1,
                                      &id,
                                      &uExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 3;
            ierr = PetscDSAddBoundary(prob,
                                      DM_BC_ESSENTIAL,
                                      "top wall temp",
                                      "marker",
                                      TEMP,
                                      0,
                                      NULL,
                                      (void (*)(void))PetscTestingFunction::ApplySolution,
                                      (void (*)(void))PetscTestingFunction::ApplySolutionTimeDerivative,
                                      1,
                                      &id,
                                      &TExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 1;
            ierr = PetscDSAddBoundary(prob,
                                      DM_BC_ESSENTIAL,
                                      "bottom wall temp",
                                      "marker",
                                      TEMP,
                                      0,
                                      NULL,
                                      (void (*)(void))PetscTestingFunction::ApplySolution,
                                      (void (*)(void))PetscTestingFunction::ApplySolutionTimeDerivative,
                                      1,
                                      &id,
                                      &TExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 2;
            ierr = PetscDSAddBoundary(prob,
                                      DM_BC_ESSENTIAL,
                                      "right wall temp",
                                      "marker",
                                      TEMP,
                                      0,
                                      NULL,
                                      (void (*)(void))PetscTestingFunction::ApplySolution,
                                      (void (*)(void))PetscTestingFunction::ApplySolutionTimeDerivative,
                                      1,
                                      &id,
                                      &TExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            id = 4;
            ierr = PetscDSAddBoundary(prob,
                                      DM_BC_ESSENTIAL,
                                      "left wall temp",
                                      "marker",
                                      TEMP,
                                      0,
                                      NULL,
                                      (void (*)(void))PetscTestingFunction::ApplySolution,
                                      (void (*)(void))PetscTestingFunction::ApplySolutionTimeDerivative,
                                      1,
                                      &id,
                                      &TExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            // Set the exact solution
            ierr = PetscDSSetExactSolution(prob, VEL, PetscTestingFunction::ApplySolution, &uExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolution(prob, PRES, PetscTestingFunction::ApplySolution, &pExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolution(prob, TEMP, PetscTestingFunction::ApplySolution, &TExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, VEL, PetscTestingFunction::ApplySolutionTimeDerivative, &uExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, PRES, PetscTestingFunction::ApplySolutionTimeDerivative, &pExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
            ierr = PetscDSSetExactSolutionTimeDerivative(prob, TEMP, PetscTestingFunction::ApplySolutionTimeDerivative, &TExact);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        // enable aux fields
        ierr = IncompressibleFlow_EnableAuxFields(flowData);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = IncompressibleFlow_CompleteProblemSetup(flowData, ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Name the flow field
        ierr = PetscObjectSetName(((PetscObject)flowData->flowField), "Numerical Solution");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // update the source terms before each time step
        PetscTestingFunction vSource(testingParam.vSource);
        PetscTestingFunction qSource(testingParam.qSource);
        PetscTestingFunction wSource(testingParam.wSource);
        TestingAuxFieldUpdater updater;
        updater.AddField(vSource);
        updater.AddField(qSource);
        updater.AddField(wSource);

        ierr = TSSetTimeStep(ts, 0.0);  // set the initial time step to 0 for the initial UpdateSourceTerms run
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = FlowRegisterPreStep(flowData, TestingAuxFieldUpdater::UpdateSourceTerms, &updater);
        TestingAuxFieldUpdater::UpdateSourceTerms(ts, &updater);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Setup the TS
        ierr = TSSetFromOptions(ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Set initial conditions from the exact solution
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
        ierr = TSMonitorSet(ts, MonitorError, NULL, NULL);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = TSSolve(ts, flowData->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Compare the actual vs expected values
        ierr = DMTSCheckFromOptions(ts, flowData->flowField);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Cleanup
        ierr = FlowDestroy(&flowData);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = DMDestroy(&dm);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = TSDestroy(&ts);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscBagDestroy(&parameterBag);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PetscFinalize();
        exit(ierr);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    IncompressibleFlow, IncompressibleFlowDynamicSourceMMS,
    testing::Values(
        (IncompressibleFlowDynamicSourceMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d quadratic tri_p2_p1_p1",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/incompressible_2d_tri_p2_p1_p1",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                                              "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                              "-momentum_source_petscspace_degree 2 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 2"},
            .uExact = "t + x^2 + y^2, t + 2*x^2 - 2*x*y, 1.0, 1.0",
            .pExact = "x + y -1, 0.0",
            .TExact = "t + x +y, 1.0",
            .vSource = "-(1-4+1+2*y*(t+2*x^2-2*x*y)+2*x*(t+x^2+y^2)), -(1-4+1-2*x*(t+2*x^2-2*x*y)+(4*x-2*y)*(t+x^2+y^2))",
            .wSource = "-(1+2*t+3*x^2-2*x*y+y^2)",
            .qSource = ".0"},
        (IncompressibleFlowDynamicSourceMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d quadratic tri_p2_p1_p1 4 proc",
                                 .nproc = 4,
                                 .expectedOutputFile = "outputs/incompressible_2d_tri_p2_p1_p1_nproc4",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 1 -dm_distribute "
                                              "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                              "-momentum_source_petscspace_degree 2 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 2"},
            .uExact = "t + x^2 + y^2, t + 2*x^2 - 2*x*y, 1.0, 1.0",
            .pExact = "x + y -1, 0.0",
            .TExact = "t + x +y, 1.0",
            .vSource = "-(1-4+1+2*y*(t+2*x^2-2*x*y)+2*x*(t+x^2+y^2)), -(1-4+1-2*x*(t+2*x^2-2*x*y)+(4*x-2*y)*(t+x^2+y^2))",
            .wSource = "-(1+2*t+3*x^2-2*x*y+y^2)",
            .qSource = ".0"},
        (IncompressibleFlowDynamicSourceMMSParameters){
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
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                              "-momentum_source_petscspace_degree 5 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 5"},
            .uExact = "t + x^3 + y^3, t + 2*x^3 - 3*x^2*y, 1.0, 1.0",
            .pExact = "3/2 *x^2 + 3/2*y^2 -1, 0.0",
            .TExact = "t + 1/2*x^2 +1/2*y^2, 1.0",
            .vSource = "-(1+3*x + 3*y^2 * (t+2*x^3 - 3*x^2*y) + 3* x^2 *(t + x^3 + y^3) - (12*x -6*x + 6*y)),-(1-(12*x-6*y) + 3*y - 3*x^2 * (t +2*x^3 - 3*x^2*y) + (6*x^2 - 6*x*y)*(t+x^3+y^3))",
            .wSource = "-(-2 + 1 + y*(t+2*x^3-3*x^2*y) + x*(t+x^3+y^3))",
            .qSource = "0.0"}),
    [](const testing::TestParamInfo<IncompressibleFlowDynamicSourceMMSParameters> &info) { return info.param.mpiTestParameter.getTestName(); });

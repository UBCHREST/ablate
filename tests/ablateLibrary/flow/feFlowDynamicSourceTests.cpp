static char help[] =
    "Time-dependent Low Mach Flow in 2d channels with finite elements. We solve the Low Mach flow problem in a rectangular domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include <petsc.h>
#include <mathFunctions/parsedFunction.hpp>
#include <mesh/boxMesh.hpp>
#include <mesh/dmWrapper.hpp>
#include <parameters/petscOptionParameters.hpp>
#include "MpiTestFixture.hpp"
#include "flow/boundaryConditions/essential.hpp"
#include "flow/incompressibleFlow.hpp"
#include "flow/lowMachFlow.hpp"
#include "gtest/gtest.h"

// We can define them because they are the same between fe flows
#define VTEST 0
#define QTEST 1
#define WTEST 2

#define VEL 0
#define PRES 1
#define TEMP 2

using namespace ablate;
using namespace ablate::flow;

struct FEFlowDynamicSourceMMSParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::function<std::shared_ptr<ablate::flow::Flow>(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters,
                                                      std::shared_ptr<parameters::Parameters> options, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initializationAndExact,
                                                      std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                      std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields)>
        createMethod;
    std::string uExact;
    std::string uDerivativeExact;
    std::string pExact;
    std::string pDerivativeExact;
    std::string TExact;
    std::string TDerivativeExact;
    std::string vSource;
    std::string wSource;
    std::string qSource;
};

class FEFlowDynamicSourceMMSTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<FEFlowDynamicSourceMMSParameters> {
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

    // Get the flowData
    ablate::flow::Flow *flow;
    ierr = DMGetApplicationContext(dm, &flow);
    CHKERRQ(ierr);

    // get the flow to apply the completeFlowInitialization method
    flow->CompleteFlowInitialization(dm, u);
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

    PetscFunctionReturn(0);
}

TEST_P(FEFlowDynamicSourceMMSTestFixture, ShouldConvergeToExactSolution) {
    StartWithMPI
        {
            TS ts; /* timestepper */

            PetscReal t;

            // Get the testing param
            auto testingParam = GetParam();

            // initialize petsc and mpi
            PetscInitialize(argc, argv, NULL, help) >> testErrorChecker;

            // setup the ts
            TSCreate(PETSC_COMM_WORLD, &ts) >> testErrorChecker;

            // Create a simple test mesh
            auto mesh = std::make_shared<mesh::BoxMesh>("mesh", std::vector<int>{2, 2}, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0});

            TSSetDM(ts, mesh->GetDomain()) >> testErrorChecker;
            TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> testErrorChecker;

            // Setup the flow data
            // pull the parameters from the petsc options
            auto parameters = std::make_shared<ablate::parameters::PetscOptionParameters>();

            auto velocityExact = std::make_shared<mathFunctions::FieldSolution>(
                "velocity", std::make_shared<mathFunctions::ParsedFunction>(testingParam.uExact), std::make_shared<mathFunctions::ParsedFunction>(testingParam.uDerivativeExact));
            auto pressureExact = std::make_shared<mathFunctions::FieldSolution>(
                "pressure", std::make_shared<mathFunctions::ParsedFunction>(testingParam.pExact), std::make_shared<mathFunctions::ParsedFunction>(testingParam.pDerivativeExact));
            auto temperatureExact = std::make_shared<mathFunctions::FieldSolution>(
                "temperature", std::make_shared<mathFunctions::ParsedFunction>(testingParam.TExact), std::make_shared<mathFunctions::ParsedFunction>(testingParam.TDerivativeExact));

            // Create the flow object
            std::shared_ptr<ablate::flow::Flow> flowObject =
                testingParam.createMethod("testFlow",
                                          mesh,
                                          parameters,
                                          nullptr,
                                          /* initialization functions */
                                          std::vector<std::shared_ptr<mathFunctions::FieldSolution>>{velocityExact, pressureExact, temperatureExact},
                                          /* boundary conditions */
                                          std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>>{
                                              std::make_shared<boundaryConditions::Essential>("velocity",
                                                                                              "top wall velocity",
                                                                                              "marker",
                                                                                              3,
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.uExact),
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.uDerivativeExact)),
                                              std::make_shared<boundaryConditions::Essential>("velocity",
                                                                                              "bottom wall velocity",
                                                                                              "marker",
                                                                                              1,
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.uExact),
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.uDerivativeExact)),
                                              std::make_shared<boundaryConditions::Essential>("velocity",
                                                                                              "right wall velocity",
                                                                                              "marker",
                                                                                              2,
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.uExact),
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.uDerivativeExact)),
                                              std::make_shared<boundaryConditions::Essential>(
                                                  "velocity",
                                                  "left wall velocity",
                                                  "marker",
                                                  4,
                                                  std::make_shared<mathFunctions::ParsedFunction>(testingParam.uExact),
                                                  std::make_shared<mathFunctions::ParsedFunction>(testingParam.uDerivativeExact)),
                                              std::make_shared<boundaryConditions::Essential>("temperature",
                                                                                              "top wall temp",
                                                                                              "marker",
                                                                                              3,
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.TExact),
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.TDerivativeExact)),
                                              std::make_shared<boundaryConditions::Essential>("temperature",
                                                                                              "bottom wall temp",
                                                                                              "marker",
                                                                                              1,
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.TExact),
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.TDerivativeExact)),
                                              std::make_shared<boundaryConditions::Essential>(
                                                  "temperature",
                                                  "right wall temp",
                                                  "marker",
                                                  2,
                                                  std::make_shared<mathFunctions::ParsedFunction>(testingParam.TExact),
                                                  std::make_shared<mathFunctions::ParsedFunction>(testingParam.TDerivativeExact)),
                                              std::make_shared<boundaryConditions::Essential>("temperature",
                                                                                              "left wall temp",
                                                                                              "marker",
                                                                                              4,
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.TExact),
                                                                                              std::make_shared<mathFunctions::ParsedFunction>(testingParam.TDerivativeExact))},
                                          /* aux field updates */
                                          std::vector<std::shared_ptr<mathFunctions::FieldSolution>>{
                                              std::make_shared<mathFunctions::FieldSolution>("momentum_source", std::make_shared<mathFunctions::ParsedFunction>(testingParam.vSource)),
                                              std::make_shared<mathFunctions::FieldSolution>("mass_source", std::make_shared<mathFunctions::ParsedFunction>(testingParam.qSource)),
                                              std::make_shared<mathFunctions::FieldSolution>("energy_source", std::make_shared<mathFunctions::ParsedFunction>(testingParam.wSource))});

            flowObject->CompleteProblemSetup(ts);

            // Name the flow field
            PetscObjectSetName(((PetscObject)flowObject->GetSolutionVector()), "Numerical Solution") >> testErrorChecker;
            VecSetOptionsPrefix(flowObject->GetSolutionVector(), "num_sol_") >> testErrorChecker;

            // set the initial time step to 0 for the initial UpdateSourceTerms run
            TSSetTimeStep(ts, 0.0) >> testErrorChecker;
            ablate::flow::Flow::UpdateAuxFields(ts, *flowObject);

            // Setup the TS
            TSSetFromOptions(ts) >> testErrorChecker;

            // Set initial conditions from the exact solution
            TSSetComputeInitialCondition(ts, SetInitialConditions) >> testErrorChecker;
            SetInitialConditions(ts, flowObject->GetSolutionVector()) >> testErrorChecker;
            TSGetTime(ts, &t) >> testErrorChecker;

            DMSetOutputSequenceNumber(flowObject->GetDM(), 0, t) >> testErrorChecker;
            DMTSCheckFromOptions(ts, flowObject->GetSolutionVector()) >> testErrorChecker;
            TSMonitorSet(ts, MonitorError, NULL, NULL) >> testErrorChecker;

            TSSolve(ts, flowObject->GetSolutionVector()) >> testErrorChecker;

            // Compare the actual vs expected values
            DMTSCheckFromOptions(ts, flowObject->GetSolutionVector()) >> testErrorChecker;

            // Cleanup
            TSDestroy(&ts) >> testErrorChecker;
        }
        exit(PetscFinalize());
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    FEFlow, FEFlowDynamicSourceMMSTestFixture,
    testing::Values(
        (FEFlowDynamicSourceMMSParameters){
            .mpiTestParameter = {.testName = "lowMach 2d quadratic tri_p3_p2_p2",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/flow/lowMach_dynamicSource_2d_tri_p3_p2_p2",
                                 .arguments = "-dm_plex_separate_marker  -dm_refine 0 "
                                              "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 -ksp_type dgmres -ksp_gmres_restart 10 "
                                              "-ksp_rtol 1.0e-9 -ksp_atol 1.0e-12 -ksp_error_if_not_converged -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 "
                                              "-pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_ksp_atol 1e-12 -fieldsplit_pressure_pc_type jacobi "
                                              "-dmts_check -1 -snes_linesearch_type basic "
                                              "-gravityDirection 1 "
                                              "-momentum_source_petscspace_degree 8 -mass_source_petscspace_degree 8  -energy_source_petscspace_degree 8"},
            .createMethod =
                [](auto name, auto mesh, auto parameters, auto options, auto initializationAndExact, auto boundaryConditions, auto auxiliaryFields) {
                    return std::make_shared<ablate::flow::LowMachFlow>(name, mesh, parameters, options, initializationAndExact, boundaryConditions, auxiliaryFields, initializationAndExact);
                },
            .uExact = "t + x^2 + y^2, t + 2*x^2 + 2*x*y",
            .uDerivativeExact = "1.0, 1.0",
            .pExact = "x + y -1",
            .pDerivativeExact = "0.0",
            .TExact = "t + x +y +1",
            .TDerivativeExact = "1.0",
            .vSource = "-(1-16/3+1/(1+t+x+y) + 2*y*(t+2*x^2+2*x*y)/(1+t+x+y) + 2*x*(t+x^2+y^2)/(1+t+x+y)), -(1 - 4 + 1/(1+t+x+y) + 1/(1+t+x+y) + 2*x*(t+2*x^2+2*x*y)/(1+t+x+y) + "
                       "(4*x+2*y)*(t+x^2+y^2)/(1+t+x+y))",
            .wSource = "-((1 + 2*t + 3*x^2 + 2*x*y + y^2)/(1+t+x+y))",
            .qSource = "-(-1/((1 + t + x + y)^2) + 4*x/(1+t+x+y) - (t + 2*x^2 + 2*x*y)/((1 + t + x + y)^2) - (t + x^2 + y^2)/((1 + t + x + y)^2))"},
        (FEFlowDynamicSourceMMSParameters){
            .mpiTestParameter = {.testName = "lowMach 2d cubic tri_p3_p2_p2",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/flow/lowMach_dynamicSource_2d_cubic_tri_p3_p2_p2",
                                 .arguments = "-dm_plex_separate_marker  -dm_refine 0 "
                                              "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 -ksp_type dgmres -ksp_gmres_restart 10 "
                                              "-ksp_rtol 1.0e-9 -ksp_atol 1.0e-12 -ksp_error_if_not_converged -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 "
                                              "-pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_ksp_atol 1e-12 -fieldsplit_pressure_pc_type jacobi "
                                              "-dmts_check -1 -snes_linesearch_type basic "
                                              "-gravityDirection 1 "
                                              "-momentum_source_petscspace_degree 8 -mass_source_petscspace_degree 8  -energy_source_petscspace_degree 8"},
            .createMethod =
                [](auto name, auto mesh, auto parameters, auto options, auto initializationAndExact, auto boundaryConditions, auto auxiliaryFields) {
                    return std::make_shared<ablate::flow::LowMachFlow>(name, mesh, parameters, options, initializationAndExact, boundaryConditions, auxiliaryFields, initializationAndExact);
                },
            .uExact = "t + x^3 + y^3, t + 2*x^3 + 3*x^2*y",
            .uDerivativeExact = "1.0, 1.0",
            .pExact = "3/2*x^2 + 3/2*y^2 -1.125",
            .pDerivativeExact = "0.0",
            .TExact = "t + .5*x^2 +.5*y^2 +1",
            .TDerivativeExact = "1.0",
            .vSource = "-(3*x + 1/(1 + t + x^2/2 + y^2/2) + (3 * y^2*(t + 2*x^3 + 3*x^2*y))/(1 + t + x^2/2 + y^2/2) + (3*x^2*(t + x^3 + y^3))/(1 + t + x^2/2 + y^2/2) - (4*x + 1*(6*x + 6*y))), -(3*y "
                       "- ((12*x + 6*y)) + 1/((1 + t + x^2/2 + y^2/2)) + 1/(1 + t + x^2/2 + y^2/2) + (3 * x^2*(t + 2*x^3 + 3*x^2*y))/(1 + t + x^2/2 + y^2/2) + ((6*x^2 + 6*x*y)*(t + x^3 + y^3))/(1 + "
                       "t + x^2/2 + y^2/2))",
            .wSource = "-(-2 + ((1 + y*(t + 2*x^3 + 3*x^2*y) + x*(t + x^3 + y^3)))/( 1 + t + x^2/2 + y^2/2))",
            .qSource =
                "-(-(1/(1 + t + x^2/2 + y^2/2)^2) - ( y * (t + 2*x^3 + 3*x^2*y))/(1 + t + x^2/2 + y^2/2)^2  + (6 * x^2)/(1 + t + x^2/2 + y^2/2) - (x * (t + x^3 + y^3))/(1 + t + x^2/2 + y^2/2)^2)"},
        (FEFlowDynamicSourceMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d quadratic tri_p2_p1_p1",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/flow/incompressible_2d_tri_p2_p1_p1",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                                              "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                              "-momentum_source_petscspace_degree 2 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 2"},
            .createMethod =
                [](auto name, auto mesh, auto parameters, auto options, auto initializationAndExact, auto boundaryConditions, auto auxiliaryFields) {
                    return std::make_shared<ablate::flow::IncompressibleFlow>(name, mesh, parameters, options, initializationAndExact, boundaryConditions, auxiliaryFields, initializationAndExact);
                },
            .uExact = "t + x^2 + y^2, t + 2*x^2 - 2*x*y",
            .uDerivativeExact = "1.0, 1.0",
            .pExact = "x + y -1",
            .pDerivativeExact = "0.0",
            .TExact = "t + x +y",
            .TDerivativeExact = "1.0",
            .vSource = "-(1-4+1+2*y*(t+2*x^2-2*x*y)+2*x*(t+x^2+y^2)), -(1-4+1-2*x*(t+2*x^2-2*x*y)+(4*x-2*y)*(t+x^2+y^2))",
            .wSource = "-(1+2*t+3*x^2-2*x*y+y^2)",
            .qSource = ".0"},
        (FEFlowDynamicSourceMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d quadratic tri_p2_p1_p1 4 proc",
                                 .nproc = 4,
                                 .expectedOutputFile = "outputs/flow/incompressible_2d_tri_p2_p1_p1_nproc4",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 1 -dm_distribute "
                                              "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                              "-momentum_source_petscspace_degree 2 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 2"},
            .createMethod =
                [](auto name, auto mesh, auto parameters, auto options, auto initializationAndExact, auto boundaryConditions, auto auxiliaryFields) {
                    return std::make_shared<ablate::flow::IncompressibleFlow>(name, mesh, parameters, options, initializationAndExact, boundaryConditions, auxiliaryFields, initializationAndExact);
                },
            .uExact = "t + x^2 + y^2, t + 2*x^2 - 2*x*y",
            .uDerivativeExact = "1.0, 1.0",
            .pExact = "x + y -1",
            .pDerivativeExact = "0.0",
            .TExact = "t + x +y",
            .TDerivativeExact = "1.0",
            .vSource = "-(1-4+1+2*y*(t+2*x^2-2*x*y)+2*x*(t+x^2+y^2)), -(1-4+1-2*x*(t+2*x^2-2*x*y)+(4*x-2*y)*(t+x^2+y^2))",
            .wSource = "-(1+2*t+3*x^2-2*x*y+y^2)",
            .qSource = ".0"},
        (FEFlowDynamicSourceMMSParameters){
            .mpiTestParameter = {.testName = "incompressible 2d cubic tri_p3_p2_p2",
                                 .nproc = 1,
                                 .expectedOutputFile = "outputs/flow/incompressible_2d_tri_p3_p2_p2",
                                 .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                                              "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                                              "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                              "-snes_convergence_test correct_pressure "
                                              "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                              "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                              "-fieldsplit_0_pc_type lu "
                                              "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                              "-momentum_source_petscspace_degree 5 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 5"},
            .createMethod =
                [](auto name, auto mesh, auto parameters, auto options, auto initializationAndExact, auto boundaryConditions, auto auxiliaryFields) {
                    return std::make_shared<ablate::flow::IncompressibleFlow>(name, mesh, parameters, options, initializationAndExact, boundaryConditions, auxiliaryFields, initializationAndExact);
                },
            .uExact = "t + x^3 + y^3, t + 2*x^3 - 3*x^2*y",
            .uDerivativeExact = "1.0, 1.0",
            .pExact = "3/2 *x^2 + 3/2*y^2 -1",
            .pDerivativeExact = "0.0",
            .TExact = "t + 1/2*x^2 +1/2*y^2",
            .TDerivativeExact = "1.0",
            .vSource = "-(1+3*x + 3*y^2 * (t+2*x^3 - 3*x^2*y) + 3* x^2 *(t + x^3 + y^3) - (12*x -6*x + 6*y)),-(1-(12*x-6*y) + 3*y - 3*x^2 * (t +2*x^3 - 3*x^2*y) + (6*x^2 - 6*x*y)*(t+x^3+y^3))",
            .wSource = "-(-2 + 1 + y*(t+2*x^3-3*x^2*y) + x*(t+x^3+y^3))",
            .qSource = "0.0"}),
    [](const testing::TestParamInfo<FEFlowDynamicSourceMMSParameters> &info) { return info.param.mpiTestParameter.getTestName(); });
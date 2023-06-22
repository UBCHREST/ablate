static char help[] =
    "Time-dependent Low Mach Flow in 2d channels with finite elements. We solve the Low Mach flow problem in a rectangular domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include <petsc.h>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "environment/runEnvironment.hpp"
#include "finiteElement/boundaryConditions/essential.hpp"
#include "finiteElement/incompressibleFlowSolver.hpp"
#include "finiteElement/lowMachFlowFields.hpp"
#include "finiteElement/lowMachFlowSolver.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/simpleFormula.hpp"
#include "parameters/petscOptionParameters.hpp"
#include "utilities/petscUtilities.hpp"

// We can define them because they are the same between fe flows
#define VTEST 0
#define QTEST 1
#define WTEST 2

using namespace ablate;
using namespace ablate::finiteElement;

struct FEFlowDynamicSourceMMSParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::function<std::shared_ptr<ablate::finiteElement::FiniteElementSolver>(std::string name, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
                                                                              std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                                              std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields)>
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

    PetscFunctionBegin;
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(TSGetTime(ts, &t));

    // This function Tags the u vector as the exact solution.  We need to copy the values to prevent this.
    Vec e;
    PetscCall(VecDuplicate(u, &e));
    PetscCall(DMComputeExactSolution(dm, t, e, NULL));
    PetscCall(VecCopy(e, u));
    PetscCall(VecDestroy(&e));

    // Get the flowData
    ablate::finiteElement::FiniteElementSolver *flow;
    PetscCall(DMGetApplicationContext(dm, &flow));

    // get the flow to apply the completeFlowInitialization method
    flow->CompleteFlowInitialization(dm, u);

    PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void *ctxs[3];
    DM dm;
    PetscDS ds;
    PetscReal ferrors[3];
    PetscInt f;

    PetscFunctionBeginUser;
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMGetDS(dm, &ds));

    for (f = 0; f < 3; ++f) {
        PetscCallAbort(PETSC_COMM_WORLD, PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]));
    }
    PetscCallAbort(PETSC_COMM_WORLD, DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors));
    PetscCallAbort(
        PETSC_COMM_WORLD,
        PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int)step, (double)crtime, (double)ferrors[0], (double)ferrors[1], (double)ferrors[2]));

    PetscFunctionReturn(0);
}

TEST_P(FEFlowDynamicSourceMMSTestFixture, ShouldConvergeToExactSolution) {
    StartWithMPI
        {
            PetscReal t;

            // Get the testing param
            auto testingParam = GetParam();

            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize(help);

            // setup the required fields for the flow
            std::vector<std::shared_ptr<domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteElement::LowMachFlowFields>(ablate::domain::Region::ENTIREDOMAIN, true)};

            // Create a simple test mesh
            auto mesh = std::make_shared<domain::BoxMesh>(
                "mesh", fieldDescriptors, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{}, std::vector<int>{2, 2}, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0});

            // pull the parameters from the petsc options
            auto parameters = std::make_shared<ablate::parameters::PetscOptionParameters>();

            auto velocityExact = std::make_shared<mathFunctions::FieldFunction>(
                "velocity", std::make_shared<mathFunctions::SimpleFormula>(testingParam.uExact), std::make_shared<mathFunctions::SimpleFormula>(testingParam.uDerivativeExact));
            auto pressureExact = std::make_shared<mathFunctions::FieldFunction>(
                "pressure", std::make_shared<mathFunctions::SimpleFormula>(testingParam.pExact), std::make_shared<mathFunctions::SimpleFormula>(testingParam.pDerivativeExact));
            auto temperatureExact = std::make_shared<mathFunctions::FieldFunction>(
                "temperature", std::make_shared<mathFunctions::SimpleFormula>(testingParam.TExact), std::make_shared<mathFunctions::SimpleFormula>(testingParam.TDerivativeExact));

            // create a time stepper
            auto timeStepper = ablate::solver::TimeStepper(mesh,
                                                           nullptr,
                                                           {},
                                                           std::make_shared<ablate::domain::Initializer>(velocityExact, pressureExact, temperatureExact),
                                                           std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{velocityExact, pressureExact, temperatureExact});

            // Create the flow object
            std::shared_ptr<ablate::finiteElement::FiniteElementSolver> flowObject =
                testingParam.createMethod("testFlow",
                                          nullptr,
                                          parameters,
                                          /* boundary conditions */
                                          std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>>{std::make_shared<boundaryConditions::Essential>("top wall velocity", 3, velocityExact),
                                                                                                              std::make_shared<boundaryConditions::Essential>("bottom wall velocity", 1, velocityExact),
                                                                                                              std::make_shared<boundaryConditions::Essential>("right wall velocity", 2, velocityExact),
                                                                                                              std::make_shared<boundaryConditions::Essential>("left wall velocity", 4, velocityExact),
                                                                                                              std::make_shared<boundaryConditions::Essential>("top wall temp", 3, temperatureExact),
                                                                                                              std::make_shared<boundaryConditions::Essential>("bottom wall temp", 1, temperatureExact),
                                                                                                              std::make_shared<boundaryConditions::Essential>("right wall temp", 2, temperatureExact),
                                                                                                              std::make_shared<boundaryConditions::Essential>("left wall temp", 4, temperatureExact)},
                                          /* aux field updates */
                                          std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{
                                              std::make_shared<mathFunctions::FieldFunction>("momentum_source", std::make_shared<mathFunctions::SimpleFormula>(testingParam.vSource)),
                                              std::make_shared<mathFunctions::FieldFunction>("mass_source", std::make_shared<mathFunctions::SimpleFormula>(testingParam.qSource)),
                                              std::make_shared<mathFunctions::FieldFunction>("energy_source", std::make_shared<mathFunctions::SimpleFormula>(testingParam.wSource))});

            timeStepper.Register(flowObject);

            // Set up the solvers
            timeStepper.Initialize();

            // set the initial time step to 0 for the initial UpdateSourceTerms run
            TSSetTimeStep(timeStepper.GetTS(), 0.0) >> testErrorChecker;
            ablate::finiteElement::FiniteElementSolver::UpdateAuxFields(timeStepper.GetTS(), *flowObject);

            // Setup the TS
            TSSetFromOptions(timeStepper.GetTS()) >> testErrorChecker;

            // Set initial conditions from the exact solution
            DMSetApplicationContext(mesh->GetDM(), flowObject.get());
            TSSetComputeInitialCondition(timeStepper.GetTS(), SetInitialConditions) >> testErrorChecker;
            SetInitialConditions(timeStepper.GetTS(), mesh->GetSolutionVector()) >> testErrorChecker;
            TSGetTime(timeStepper.GetTS(), &t) >> testErrorChecker;

            DMSetOutputSequenceNumber(mesh->GetDM(), 0, t) >> testErrorChecker;
            DMTSCheckFromOptions(timeStepper.GetTS(), mesh->GetSolutionVector()) >> testErrorChecker;
            TSMonitorSet(timeStepper.GetTS(), MonitorError, NULL, NULL) >> testErrorChecker;

            // Solve in time
            timeStepper.Solve();

            // Compare the actual vs expected values
            DMTSCheckFromOptions(timeStepper.GetTS(), mesh->GetSolutionVector()) >> testErrorChecker;
        }
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    FEFlow, FEFlowDynamicSourceMMSTestFixture,
    testing::Values(
        (FEFlowDynamicSourceMMSParameters){
            .mpiTestParameter =
                testingResources::MpiTestParameter("lowMach 2d quadratic tri_p3_p2_p2", 1,
                                                   "-dm_plex_separate_marker  -dm_refine 0 "
                                                   "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                                                   "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 -ksp_type dgmres -ksp_gmres_restart 10 "
                                                   "-ksp_rtol 1.0e-9 -ksp_atol 1.0e-12 -ksp_error_if_not_converged -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 "
                                                   "-pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                   "-fieldsplit_0_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_ksp_atol 1e-12 -fieldsplit_pressure_pc_type jacobi "
                                                   "-dmts_check -1 -snes_linesearch_type basic "
                                                   "-gravityDirection 1 "
                                                   "-momentum_source_petscspace_degree 8 -mass_source_petscspace_degree 8  -energy_source_petscspace_degree 8",
                                                   "outputs/finiteElement/lowMach_dynamicSource_2d_tri_p3_p2_p2"),
            .createMethod =
                [](auto name, auto parameters, auto options, auto boundaryConditions, auto auxiliaryFields) {
                    return std::make_shared<ablate::finiteElement::LowMachFlowSolver>(name, ablate::domain::Region::ENTIREDOMAIN, parameters, options, boundaryConditions, auxiliaryFields);
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
            .mpiTestParameter =
                testingResources::MpiTestParameter("lowMach 2d cubic tri_p3_p2_p2", 1,
                                                   "-dm_plex_separate_marker  -dm_refine 0 "
                                                   "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                                                   "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 -ksp_type dgmres -ksp_gmres_restart 10 "
                                                   "-ksp_rtol 1.0e-9 -ksp_atol 1.0e-12 -ksp_error_if_not_converged -pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 "
                                                   "-pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                   "-fieldsplit_0_pc_type lu -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_ksp_atol 1e-12 -fieldsplit_pressure_pc_type jacobi "
                                                   "-dmts_check -1 -snes_linesearch_type basic "
                                                   "-gravityDirection 1 "
                                                   "-momentum_source_petscspace_degree 8 -mass_source_petscspace_degree 8  -energy_source_petscspace_degree 8",
                                                   "outputs/finiteElement/lowMach_dynamicSource_2d_cubic_tri_p3_p2_p2"),
            .createMethod =
                [](auto name, auto parameters, auto options, auto boundaryConditions, auto auxiliaryFields) {
                    return std::make_shared<ablate::finiteElement::LowMachFlowSolver>(name, ablate::domain::Region::ENTIREDOMAIN, parameters, options, boundaryConditions, auxiliaryFields);
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
        (FEFlowDynamicSourceMMSParameters){.mpiTestParameter = testingResources::MpiTestParameter(
                                               "incompressible 2d quadratic tri_p2_p1_p1", 1,
                                               "-dm_plex_separate_marker -dm_refine 0 "
                                               "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                               "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                               "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                               "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                               "-fieldsplit_0_pc_type lu "
                                               "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                               "-momentum_source_petscspace_degree 2 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 2",
                                               "outputs/finiteElement/incompressible_2d_tri_p2_p1_p1"),
                                           .createMethod =
                                               [](auto name, auto parameters, auto options, auto boundaryConditions, auto auxiliaryFields) {
                                                   return std::make_shared<ablate::finiteElement::IncompressibleFlowSolver>(
                                                       name, ablate::domain::Region::ENTIREDOMAIN, parameters, options, boundaryConditions, auxiliaryFields);
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
        (FEFlowDynamicSourceMMSParameters){.mpiTestParameter = testingResources::MpiTestParameter(
                                               "incompressible 2d quadratic tri_p2_p1_p1 4 proc", 4,
                                               "-dm_plex_separate_marker -dm_refine 1 -dm_distribute "
                                               "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                               "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                               "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                               "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                               "-fieldsplit_0_pc_type lu "
                                               "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                               "-momentum_source_petscspace_degree 2 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 2",
                                               "outputs/finiteElement/incompressible_2d_tri_p2_p1_p1_nproc4"),
                                           .createMethod =
                                               [](auto name, auto parameters, auto options, auto boundaryConditions, auto auxiliaryFields) {
                                                   return std::make_shared<ablate::finiteElement::IncompressibleFlowSolver>(
                                                       name, ablate::domain::Region::ENTIREDOMAIN, parameters, options, boundaryConditions, auxiliaryFields);
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
            .mpiTestParameter =
                testingResources::MpiTestParameter("incompressible 2d cubic tri_p3_p2_p2", 1,
                                                   "-dm_plex_separate_marker -dm_refine 0 "
                                                   "-vel_petscspace_degree 3 -pres_petscspace_degree 2 -temp_petscspace_degree 2 "
                                                   "-dmts_check .001 -ts_max_steps 4 -ts_dt 0.1 "
                                                   "-snes_convergence_test correct_pressure "
                                                   "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                                   "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                                   "-fieldsplit_0_pc_type lu "
                                                   "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi "
                                                   "-momentum_source_petscspace_degree 5 -mass_source_petscspace_degree 1 -energy_source_petscspace_degree 5",
                                                   "outputs/finiteElement/incompressible_2d_cubic_p3_p2_p2"),
            .createMethod =
                [](auto name, auto parameters, auto options, auto boundaryConditions, auto auxiliaryFields) {
                    return std::make_shared<ablate::finiteElement::IncompressibleFlowSolver>(name, domain::Region::ENTIREDOMAIN, parameters, options, boundaryConditions, auxiliaryFields);
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
#include <petsc.h>
#include <cmath>
#include <convergenceTester.hpp>
#include <eos/mockEOS.hpp>
#include <eos/transport/constant.hpp>
#include <flow/boundaryConditions/essentialGhost.hpp>
#include <flow/processes/eulerDiffusion.hpp>
#include <flow/processes/speciesDiffusion.hpp>
#include <map>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include <mesh/boxMesh.hpp>
#include <monitors/solutionErrorMonitor.hpp>
#include <solve/timeStepper.hpp>
#include <utilities/petscOptions.hpp>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "eos/mockEOS.hpp"
#include "flow/boundaryConditions/ghost.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

typedef struct {
    PetscInt dim;
    PetscReal L;
    PetscReal diff;
    PetscReal rho;
} InputParameters;

struct CompressibleSpeciesDiffusionTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    InputParameters parameters;
    PetscInt initialNx;
    int levels;
    std::vector<PetscReal> expectedL2Convergence;
    std::vector<PetscReal> expectedLInfConvergence;
};

using namespace ablate;

class CompressibleFlowSpeciesDiffusionTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<CompressibleSpeciesDiffusionTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

///////////////////////////////////////////////////////////////
const static PetscReal speciesSensibleEnthalpy[3] = {1000.0, 2000.0, 3000.0};
const static std::vector<std::string> species = {"sp0", "sp1", "sp2"};
static PetscErrorCode MockTemperatureFunction(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
    *T = NAN;
    return 0;
}

static PetscErrorCode MockSpeciesSensibleEnthalpyFunction(PetscReal T, PetscReal* hi, void* ctx) {
    for (std::size_t s = 0; s < species.size(); s++) {
        hi[s] = speciesSensibleEnthalpy[s];
    }
    return 0;
}

////////////////////////////////////
static PetscErrorCode ComputeDensityYiExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar* yi, void* ctx) {
    PetscFunctionBeginUser;
    InputParameters* parameters = (InputParameters*)ctx;
    PetscReal yiInit = 0.5;
    PetscReal yi0 = 0.0;
    for (PetscReal n = 1; n < 2000; n++) {
        PetscReal Bn = -yiInit * 2.0 * (-1.0 + PetscPowReal(-1.0, n)) / (n * PETSC_PI);
        yi0 += Bn * PetscSinReal(n * PETSC_PI * xyz[0] / parameters->L) * PetscExpReal(-n * n * PETSC_PI * PETSC_PI * parameters->diff * time / (PetscSqr(parameters->L)));
    }

    yi[0] = yi0 * parameters->rho;
    yi[1] = (1.0 - .5 - yi0) * parameters->rho;
    yi[2] = 0.5 * parameters->rho;
    PetscFunctionReturn(0);
}

/**
 * Computes the euler exact assuming constant density, no velocity, and rho*e assuming that e is a sum o yi*hi
 */
static PetscErrorCode ComputeEulerExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar* euler, void* ctx) {
    PetscFunctionBeginUser;
    InputParameters* parameters = (InputParameters*)ctx;

    euler[0] = parameters->rho;
    euler[1] = 0.0;
    euler[2] = 0.0;
    euler[3] = 0.0;

    // compute the current yi
    std::vector<PetscReal> rhoYi(species.size());
    ComputeDensityYiExact(dim, time, xyz, Nf, &rhoYi[0], ctx);
    for (std::size_t s = 0; s < rhoYi.size(); s++) {
        euler[1] += rhoYi[s] * speciesSensibleEnthalpy[s];
    }

    PetscFunctionReturn(0);
}
TEST_P(CompressibleFlowSpeciesDiffusionTestFixture, ShouldConvergeToExactSolution) {
    StartWithMPI
        PetscErrorCode ierr;

        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, "HELP") >> testErrorChecker;

        // keep track of history
        testingResources::ConvergenceTester l2History("l2");
        testingResources::ConvergenceTester lInfHistory("lInf");

        // get the input params
        InputParameters parameters = GetParam().parameters;

        // March over each level
        for (PetscInt l = 1; l <= GetParam().levels; l++) {
            PetscPrintf(PETSC_COMM_WORLD, "Running Calculation at Level %d\n", l);

            // setup any global arguments
            ablate::utilities::PetscOptionsUtils::Set({{"dm_plex_separate_marker", ""}, {"automaticTimeStepCalculator", "off"}, {"petsclimiter_type", "none"}});

            // create a time stepper
            auto timeStepper = ablate::solve::TimeStepper("timeStepper", {{"ts_dt", "5.e-01"}, {"ts_type", "rk"}, {"ts_max_time", "15.0"}, {"ts_adapt_type", "none"}});

            PetscInt initialNx = GetParam().initialNx;
            auto mesh = std::make_shared<ablate::mesh::BoxMesh>("simpleMesh",
                                                                std::vector<int>{(int)initialNx, (int)initialNx},
                                                                std::vector<double>{0.0, 0.0},
                                                                std::vector<double>{parameters.L, parameters.L},
                                                                std::vector<std::string>{"NONE", "PERIODIC"} /*boundary*/,
                                                                false /*simplex*/,
                                                                std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                                                    {"dm_refine", std::to_string(l)},
                                                                    {"dm_distribute", ""},
                                                                }));

            // setup a flow parameters
            auto flowParameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{});
            auto transportModel = std::make_shared<ablate::eos::transport::Constant>(0.0, 0.0, parameters.diff);
            auto petscFlowOptions = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"yipetscfv_type", "leastsquares"}});

            // create an eos with three species
            auto eosParameters = std::make_shared<ablate::parameters::MapParameters>();

            // create a mock eos
            std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
            EXPECT_CALL(*eos, GetSpecies()).Times(::testing::AtLeast(1)).WillRepeatedly(::testing::ReturnRef(species));
            EXPECT_CALL(*eos, GetComputeTemperatureFunction()).Times(::testing::Exactly(1)).WillOnce(::testing::Return(MockTemperatureFunction));
            EXPECT_CALL(*eos, GetComputeTemperatureContext()).Times(::testing::Exactly(1));
            EXPECT_CALL(*eos, GetComputeSpeciesSensibleEnthalpyFunction()).Times(::testing::Exactly(1)).WillOnce(::testing::Return(MockSpeciesSensibleEnthalpyFunction));
            EXPECT_CALL(*eos, GetComputeSpeciesSensibleEnthalpyContext()).Times(::testing::Exactly(1));

            // create a constant density field
            auto eulerExact = mathFunctions::Create(ComputeEulerExact, &parameters);
            auto eulerExactField = std::make_shared<mathFunctions::FieldFunction>("euler", eulerExact);

            // Create the yi field solutions
            auto yiExact = ablate::mathFunctions::Create(ComputeDensityYiExact, &parameters);
            auto yiExactField = std::make_shared<mathFunctions::FieldFunction>("densityYi", yiExact);

            auto boundaryConditions =
                std::vector<std::shared_ptr<flow::boundaryConditions::BoundaryCondition>>{std::make_shared<flow::boundaryConditions::EssentialGhost>("walls", std::vector<int>{4, 2}, eulerExactField),
                                                                                          std::make_shared<flow::boundaryConditions::EssentialGhost>("left", std::vector<int>{4}, yiExactField),
                                                                                          std::make_shared<flow::boundaryConditions::EssentialGhost>("right", std::vector<int>{2}, yiExactField)};

            auto flowProcesses = std::vector<std::shared_ptr<ablate::flow::processes::FlowProcess>>{
                std::make_shared<ablate::flow::processes::SpeciesDiffusion>(eos, transportModel),
            };

            auto flowObject = std::make_shared<ablate::flow::FVFlow>(
                "testFlow",
                mesh,
                flowParameters,
                std::vector<ablate::flow::FlowFieldDescriptor>{
                    {.fieldName = "euler", .fieldPrefix = "euler", .components = 2 + mesh->GetDimensions(), .fieldType = ablate::flow::FieldType::FV},
                    {
                        .fieldName = "densityYi",
                        .fieldPrefix = "densityYi",
                        .components = (PetscInt)eos->GetSpecies().size(),
                        .fieldType = ablate::flow::FieldType::FV,
                        .componentNames = eos->GetSpecies(),
                    },
                    {.solutionField = false, .fieldName = "yi", .fieldPrefix = "yi", .components = (PetscInt)eos->GetSpecies().size(), .fieldType = ablate::flow::FieldType::FV}},
                flowProcesses,
                petscFlowOptions /*options*/,
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{eulerExactField, yiExactField} /*initialization*/,
                boundaryConditions /*boundary conditions*/,
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{},
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{eulerExactField, yiExactField});

            flowObject->SetupSolve(timeStepper.GetTS());

            // run
            timeStepper.Solve(flowObject);

            // Get the L2 and LInf norms
            std::vector<PetscReal> l2Norm = ablate::monitors::SolutionErrorMonitor(ablate::monitors::SolutionErrorMonitor::Scope::COMPONENT, ablate::monitors::SolutionErrorMonitor::Norm::L2_NORM)
                                                .ComputeError(timeStepper.GetTS(), timeStepper.GetTime(), flowObject->GetSolutionVector());
            std::vector<PetscReal> lInfNorm = ablate::monitors::SolutionErrorMonitor(ablate::monitors::SolutionErrorMonitor::Scope::COMPONENT, ablate::monitors::SolutionErrorMonitor::Norm::LINF)
                                                  .ComputeError(timeStepper.GetTS(), timeStepper.GetTime(), flowObject->GetSolutionVector());

            // print the results to help with debug
            const PetscReal h = parameters.L / (initialNx * PetscPowInt(2.0, l));
            l2History.Record(h, l2Norm);
            lInfHistory.Record(h, lInfNorm);
        }

        std::string l2Message;
        if (!l2History.CompareConvergenceRate(GetParam().expectedL2Convergence, l2Message)) {
            FAIL() << l2Message;
        }

        std::string lInfMessage;
        if (!lInfHistory.CompareConvergenceRate(GetParam().expectedLInfConvergence, lInfMessage)) {
            FAIL() << lInfMessage;
        }

        ierr = PetscFinalize();
        exit(ierr);

    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(CompressibleFlow, CompressibleFlowSpeciesDiffusionTestFixture,
                         testing::Values((CompressibleSpeciesDiffusionTestParameters){.mpiTestParameter = {.testName = "species diffusion mpi 1", .nproc = 1, .arguments = ""},
                                                                                      .parameters = {.dim = 2, .L = 0.1, .diff = 1.0E-5, .rho = 1.0},
                                                                                      .initialNx = 3,
                                                                                      .levels = 3,
                                                                                      .expectedL2Convergence = {NAN, 1.8, NAN, NAN, 1.8, 1.8, NAN},
                                                                                      .expectedLInfConvergence = {NAN, 1.0, NAN, NAN, 1.0, 1.0, NAN}},
                                         (CompressibleSpeciesDiffusionTestParameters){.mpiTestParameter = {.testName = "species diffusion mpi 1 density 2.0", .nproc = 1, .arguments = ""},
                                                                                      .parameters = {.dim = 2, .L = 0.1, .diff = 1.0E-5, .rho = 2.0},
                                                                                      .initialNx = 3,
                                                                                      .levels = 3,
                                                                                      .expectedL2Convergence = {NAN, 1.8, NAN, NAN, 1.8, 1.8, NAN},
                                                                                      .expectedLInfConvergence = {NAN, 1.0, NAN, NAN, 1.0, 1.0, NAN}},
                                         (CompressibleSpeciesDiffusionTestParameters){.mpiTestParameter = {.testName = "species diffusion mpi 2 density 2.0", .nproc = 2, .arguments = ""},
                                                                                      .parameters = {.dim = 2, .L = 0.1, .diff = 1.0E-5, .rho = 2.0},
                                                                                      .initialNx = 3,
                                                                                      .levels = 3,
                                                                                      .expectedL2Convergence = {NAN, 1.8, NAN, NAN, 1.8, 1.8, NAN},
                                                                                      .expectedLInfConvergence = {NAN, 1.0, NAN, NAN, 1.0, 1.0, NAN}}),
                         [](const testing::TestParamInfo<CompressibleSpeciesDiffusionTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <petsc.h>
#include <cmath>
#include <map>
#include <memory>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "convergenceTester.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "environment/runEnvironment.hpp"
#include "eos/mockEOS.hpp"
#include "eos/transport/constant.hpp"
#include "finiteVolume/boundaryConditions/essentialGhost.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/compressibleFlowSolver.hpp"
#include "finiteVolume/processes/speciesTransport.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "monitors/solutionErrorMonitor.hpp"
#include "parameters/mapParameters.hpp"
#include "solver/timeStepper.hpp"
#include "utilities/petscOptions.hpp"
#include "utilities/petscUtilities.hpp"

typedef struct {
    PetscReal L;
    PetscReal diff;
    PetscReal rho;
} InputParameters;

struct CompressibleEvDiffusionTestParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    InputParameters parameters;
    PetscInt initialNx;
    int levels;
    std::vector<PetscReal> expectedL2Convergence;
    std::vector<PetscReal> expectedLInfConvergence;
};

using namespace ablate;

class CompressibleFlowEvDiffusionTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<CompressibleEvDiffusionTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

////////////////////////////////////
static PetscErrorCode ComputeDensityEVExact(PetscInt dim, PetscReal time, const PetscReal xyz[], PetscInt Nf, PetscScalar* ev, void* ctx) {
    PetscFunctionBeginUser;
    InputParameters* parameters = (InputParameters*)ctx;
    PetscReal yiInit = 0.5;
    PetscReal yi0 = 0.0;
    for (PetscReal n = 1; n < 2000; n++) {
        PetscReal Bn = -yiInit * 2.0 * (-1.0 + PetscPowReal(-1.0, n)) / (n * PETSC_PI);
        yi0 += Bn * PetscSinReal(n * PETSC_PI * xyz[0] / parameters->L) * PetscExpReal(-n * n * PETSC_PI * PETSC_PI * parameters->diff * time / (PetscSqr(parameters->L)));
    }

    ev[0] = yi0 * parameters->rho;
    ev[1] = (.5 - yi0) * parameters->rho;
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
    PetscFunctionReturn(0);
}
TEST_P(CompressibleFlowEvDiffusionTestFixture, ShouldConvergeToExactSolution) {
    StartWithMPI
        PetscErrorCode ierr;

        // initialize petsc and mpi
        ablate::environment::RunEnvironment::Initialize(argc, argv);
        ablate::utilities::PetscUtilities::Initialize();

        // keep track of history
        testingResources::ConvergenceTester l2History("l2");
        testingResources::ConvergenceTester lInfHistory("lInf");

        // get the input params
        InputParameters parameters = GetParam().parameters;

        // March over each level
        for (PetscInt l = 1; l <= GetParam().levels; l++) {
            PetscPrintf(PETSC_COMM_WORLD, "Running Calculation at Level %" PetscInt_FMT "\n", l);

            // setup any global arguments
            ablate::utilities::PetscOptionsUtils::Set({{"dm_plex_separate_marker", ""}, {"petsclimiter_type", "none"}});

            PetscInt initialNx = GetParam().initialNx;

            // create a mock eos
            std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
            auto species = std::vector<std::string>();
            EXPECT_CALL(*eos, GetSpeciesVariables()).Times(::testing::AtLeast(1)).WillRepeatedly(::testing::ReturnRef(species));
            EXPECT_CALL(*eos, GetProgressVariables()).Times(::testing::AtLeast(1)).WillRepeatedly(::testing::ReturnRef(ablate::utilities::VectorUtilities::Empty<std::string>));
            EXPECT_CALL(*eos, GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Temperature, testing::_))
                .Times(::testing::AtLeast(1))
                .WillRepeatedly(
                    ::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction([](const PetscReal conserved[], PetscReal TOld, PetscReal* T) { *T = 300; })));

            // determine required fields for finite volume compressible flow
            std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {
                std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos, std::vector<std::string>{"ev1", "ev2"})};

            auto mesh = std::make_shared<ablate::domain::BoxMesh>("simpleMesh",
                                                                  fieldDescriptors,
                                                                  std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(),
                                                                                                                                    std::make_shared<domain::modifiers::GhostBoundaryCells>()},
                                                                  std::vector<int>{(int)initialNx},
                                                                  std::vector<double>{0.0},
                                                                  std::vector<double>{parameters.L},
                                                                  std::vector<std::string>{"NONE"} /*boundary*/,
                                                                  false /*simplex*/,
                                                                  ablate::parameters::MapParameters::Create({
                                                                      {"dm_refine", std::to_string(l)},
                                                                      {"dm_distribute", ""},
                                                                  }));

            // create a constant density field
            auto eulerExact = mathFunctions::Create(ComputeEulerExact, &parameters);
            auto eulerExactField = std::make_shared<mathFunctions::FieldFunction>("euler", eulerExact);

            // Create the yi field solutions
            auto evExact = ablate::mathFunctions::Create(ComputeDensityEVExact, &parameters);
            auto evExactField = std::make_shared<mathFunctions::FieldFunction>(finiteVolume::CompressibleFlowFields::DENSITY_EV_FIELD, evExact);
            std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization{eulerExactField, evExactField};

            // create a time stepper
            auto timeStepper = ablate::solver::TimeStepper(mesh,
                                                           ablate::parameters::MapParameters::Create({{"ts_dt", "5.e-01"}, {"ts_type", "rk"}, {"ts_max_time", "15.0"}, {"ts_adapt_type", "none"}}),
                                                           {},
                                                           initialization,
                                                           std::vector<std::shared_ptr<mathFunctions::FieldFunction>>{eulerExactField, evExactField});

            // setup a flow parameters
            auto transportModel = std::make_shared<ablate::eos::transport::Constant>(0.0, 0.0, parameters.diff);
            auto petscFlowOptions = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{});

            // create an eos with three species
            auto eosParameters = std::make_shared<ablate::parameters::MapParameters>();

            auto boundaryConditions = std::vector<std::shared_ptr<finiteVolume::boundaryConditions::BoundaryCondition>>{
                std::make_shared<finiteVolume::boundaryConditions::EssentialGhost>("walls", std::vector<int>{1, 2}, eulerExactField, "", true),
                std::make_shared<finiteVolume::boundaryConditions::EssentialGhost>("left", std::vector<int>{2}, evExactField, "", true),
                std::make_shared<finiteVolume::boundaryConditions::EssentialGhost>("right", std::vector<int>{1}, evExactField, "", true)};

            auto flowObject = std::make_shared<ablate::finiteVolume::CompressibleFlowSolver>("testFlow",
                                                                                             domain::Region::ENTIREDOMAIN,
                                                                                             petscFlowOptions /*options*/,
                                                                                             eos,
                                                                                             nullptr /*options*/,
                                                                                             transportModel,
                                                                                             nullptr /*no advection */,
                                                                                             std::vector<std::shared_ptr<finiteVolume::processes::Process>>(),
                                                                                             boundaryConditions /*boundary conditions*/);

            timeStepper.Register(flowObject);

            // run
            timeStepper.Solve();

            // Get the L2 and LInf norms
            std::vector<PetscReal> l2Norm = ablate::monitors::SolutionErrorMonitor(ablate::monitors::SolutionErrorMonitor::Scope::COMPONENT, ablate::monitors::SolutionErrorMonitor::Norm::L2_NORM)
                                                .ComputeError(timeStepper.GetTS(), timeStepper.GetTime(), mesh->GetSolutionVector());
            std::vector<PetscReal> lInfNorm = ablate::monitors::SolutionErrorMonitor(ablate::monitors::SolutionErrorMonitor::Scope::COMPONENT, ablate::monitors::SolutionErrorMonitor::Norm::LINF)
                                                  .ComputeError(timeStepper.GetTS(), timeStepper.GetTime(), mesh->GetSolutionVector());

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

        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(CompressibleFlow, CompressibleFlowEvDiffusionTestFixture,
                         testing::Values((CompressibleEvDiffusionTestParameters){.mpiTestParameter = {.testName = "ev diffusion mpi 1", .nproc = 1, .arguments = ""},
                                                                                 .parameters = {.L = 0.1, .diff = 1.0E-5, .rho = 1.0},
                                                                                 .initialNx = 3,
                                                                                 .levels = 3,
                                                                                 .expectedL2Convergence = {NAN, NAN, NAN, 2.2, 2.2},
                                                                                 .expectedLInfConvergence = {NAN, NAN, NAN, 2.2, 2.2}},
                                         (CompressibleEvDiffusionTestParameters){.mpiTestParameter = {.testName = "ev diffusion mpi 1 density 2.0", .nproc = 1, .arguments = ""},
                                                                                 .parameters = {.L = 0.1, .diff = 1.0E-5, .rho = 2.0},
                                                                                 .initialNx = 3,
                                                                                 .levels = 3,
                                                                                 .expectedL2Convergence = {NAN, NAN, NAN, 2.2, 2.2},
                                                                                 .expectedLInfConvergence = {NAN, NAN, NAN, 2.2, 2.2}},
                                         (CompressibleEvDiffusionTestParameters){.mpiTestParameter = {.testName = "ev diffusion mpi 2 density 2.0", .nproc = 2, .arguments = ""},
                                                                                 .parameters = {.L = 0.1, .diff = 1.0E-5, .rho = 2.0},
                                                                                 .initialNx = 3,
                                                                                 .levels = 3,
                                                                                 .expectedL2Convergence = {NAN, NAN, NAN, 2.2, 2.2},
                                                                                 .expectedLInfConvergence = {NAN, NAN, NAN, 2.2, 2.2}}),
                         [](const testing::TestParamInfo<CompressibleEvDiffusionTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

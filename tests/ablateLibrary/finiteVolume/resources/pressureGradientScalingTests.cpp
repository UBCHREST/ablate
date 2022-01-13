#include <petsc.h>
#include <cmath>
#include <domain/dmWrapper.hpp>
#include <domain/modifiers/distributeWithGhostCells.hpp>
#include <domain/modifiers/ghostBoundaryCells.hpp>
#include <finiteVolume/compressibleFlowFields.hpp>
#include <memory>
#include <solver/directSolverTsInterface.hpp>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "domain/boxMesh.hpp"
#include "eos/perfectGas.hpp"
#include "eos/transport/constant.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowSolver.hpp"
#include "finiteVolume/fieldFunctions/compressibleFlowState.hpp"
#include "finiteVolume/fieldFunctions/euler.hpp"
#include "finiteVolume/processes/eulerTransport.hpp"
#include "finiteVolume/resources/pressureGradientScaling.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "mathFunctions/geom/sphere.hpp"
#include "parameters/mapParameters.hpp"

typedef struct {
    testingResources::MpiTestParameter mpiTestParameter;
    PetscReal alphaInit;
    PetscReal domainLength;
    PetscReal maxAlphaAllowed;
    PetscReal maxDeltaPressureFac;
    PetscReal dt;
    std::function<std::shared_ptr<ablate::mathFunctions::FieldFunction>(std::shared_ptr<ablate::eos::EOS>)> getFieldFunction;

    /** Expected Values**/
    PetscReal expectedAlpha;
    PetscReal expectedMaxMach;
} PressureGradientScalingTestParameters;

class PressureGradientScalingTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<PressureGradientScalingTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

static PetscReal Difference(PetscReal expected, PetscReal actual) { return PetscAbs(expected - actual) / expected; }

TEST_P(PressureGradientScalingTestFixture, ShouldUpdatePgsCorrectly) {
    StartWithMPI
        // arrange
        PetscErrorCode ierr;

        // initialize petsc and mpi
        PetscInitialize(argc, argv, NULL, "HELP") >> testErrorChecker;

        const auto& parameters = GetParam();

        auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287"}}));
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

        auto domain =
            std::make_shared<ablate::domain::BoxMesh>("simpleMesh",
                                                      fieldDescriptors,
                                                      std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(),
                                                                                                                        std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>()},
                                                      std::vector<int>{(int)30, (int)30},
                                                      std::vector<double>{0.0, 0.0},
                                                      std::vector<double>{1.0, 1.0},
                                                      std::vector<std::string>{} /*boundary*/,
                                                      false /*simplex*/
            );

        // Create the pgs
        auto pgs =
            std::make_shared<ablate::finiteVolume::resources::PressureGradientScaling>(eos, parameters.alphaInit, parameters.domainLength, parameters.maxAlphaAllowed, parameters.maxDeltaPressureFac);

        // Create a fV for testing
        // Make a finite volume with only a gravity
        auto fvObject = std::make_shared<ablate::finiteVolume::FiniteVolumeSolver>("testFV",
                                                                                   ablate::domain::Region::ENTIREDOMAIN,
                                                                                   nullptr /*options*/,
                                                                                   std::vector<std::shared_ptr<ablate::finiteVolume::processes::Process>>{},
                                                                                   std::vector<std::shared_ptr<ablate::finiteVolume::boundaryConditions::BoundaryCondition>>{});

        // Create a ts for testing
        TS testTs;
        TSCreate(PETSC_COMM_WORLD, &testTs) >> testErrorChecker;

        // initialize the domain/fields with the specified inputs
        domain->InitializeSubDomains(std::vector<std::shared_ptr<ablate::solver::Solver>>{fvObject},
                                     std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>>{parameters.getFieldFunction(eos)});

        // act
        pgs->UpdatePreconditioner(testTs, *fvObject);

        // assert
        ASSERT_LT(Difference(parameters.expectedAlpha, pgs->GetAlpha()), 1E-4);
        ASSERT_LT(Difference(parameters.expectedMaxMach, pgs->GetMaxMach()), 1E-4);

        // cleanup
        TSDestroy(&testTs) >> testErrorChecker;
        ierr = PetscFinalize();
        exit(ierr);

    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    PressureGradientScalingTests, PressureGradientScalingTestFixture,
    testing::Values(
        (PressureGradientScalingTestParameters){.mpiTestParameter = {.testName = "pgs test 1", .nproc = 1, .arguments = ""},
                                                .alphaInit = 5.0,
                                                .domainLength = 2.0,
                                                .maxAlphaAllowed = {} /*default value*/,
                                                .maxDeltaPressureFac = {}, /*default value*/
                                                .dt = .1,
                                                .getFieldFunction =
                                                    [](std::shared_ptr<ablate::eos::EOS> eos) {
                                                        auto flowState = std::make_shared<ablate::finiteVolume::fieldFunctions::CompressibleFlowState>(
                                                            eos, ablate::mathFunctions::Create(300.0), ablate::mathFunctions::Create(101325.0), ablate::mathFunctions::Create("1.0, 0.0"));

                                                        return std::shared_ptr<ablate::mathFunctions::FieldFunction>{std::make_shared<ablate::finiteVolume::fieldFunctions::Euler>(flowState)};
                                                    },
                                                /* expected results **/
                                                .expectedAlpha = 5.5,
                                                .expectedMaxMach = 0.00288027},
        (PressureGradientScalingTestParameters){.mpiTestParameter = {.testName = "pgs test 2", .nproc = 1, .arguments = ""},
                                                .alphaInit = 5.0,
                                                .domainLength = 2.0,
                                                .maxAlphaAllowed = {} /*default value*/,
                                                .maxDeltaPressureFac = {}, /*default value*/
                                                .dt = .1,
                                                .getFieldFunction =
                                                    [](std::shared_ptr<ablate::eos::EOS> eos) {
                                                        auto flowState = std::make_shared<ablate::finiteVolume::fieldFunctions::CompressibleFlowState>(
                                                            eos, ablate::mathFunctions::Create(300.0), ablate::mathFunctions::Create(101325.0), ablate::mathFunctions::Create("0.0, 100.0"));

                                                        return std::shared_ptr<ablate::mathFunctions::FieldFunction>{std::make_shared<ablate::finiteVolume::fieldFunctions::Euler>(flowState)};
                                                    },
                                                /* expected results **/
                                                .expectedAlpha = 2.430320,
                                                .expectedMaxMach = 0.2880277994805399},
        (PressureGradientScalingTestParameters){.mpiTestParameter = {.testName = "pgs test 3", .nproc = 1, .arguments = ""},
                                                .alphaInit = 5.0,
                                                .domainLength = 2.0,
                                                .maxAlphaAllowed = 1.5,
                                                .maxDeltaPressureFac = {}, /*default value*/
                                                .dt = .1,
                                                .getFieldFunction =
                                                    [](std::shared_ptr<ablate::eos::EOS> eos) {
                                                        auto flowState = std::make_shared<ablate::finiteVolume::fieldFunctions::CompressibleFlowState>(
                                                            eos, ablate::mathFunctions::Create(300.0), ablate::mathFunctions::Create(101325.0), ablate::mathFunctions::Create("0.0, 100.0"));

                                                        return std::shared_ptr<ablate::mathFunctions::FieldFunction>{std::make_shared<ablate::finiteVolume::fieldFunctions::Euler>(flowState)};
                                                    },
                                                /* expected results **/
                                                .expectedAlpha = 1.5,
                                                .expectedMaxMach = 0.2880277994805399},
        (PressureGradientScalingTestParameters){
            .mpiTestParameter = {.testName = "pgs test 4", .nproc = 2, .arguments = ""},
            .alphaInit = 5.0,
            .domainLength = 2.0,
            .maxAlphaAllowed = {} /*default value*/,
            .maxDeltaPressureFac = {}, /*default value*/
            .dt = .1,
            .getFieldFunction =
                [](std::shared_ptr<ablate::eos::EOS> eos) {
                    auto flowState = std::make_shared<ablate::finiteVolume::fieldFunctions::CompressibleFlowState>(
                        eos, ablate::mathFunctions::Create(300.0), ablate::mathFunctions::Create("101325.0 + x*y*10000"), ablate::mathFunctions::Create("10.0, 10.0"));

                    return std::shared_ptr<ablate::mathFunctions::FieldFunction>{std::make_shared<ablate::finiteVolume::fieldFunctions::Euler>(flowState)};
                },
            /* expected results **/
            .expectedAlpha = 3.7061,
            .expectedMaxMach = 0.040733},
        (PressureGradientScalingTestParameters){
            .mpiTestParameter = {.testName = "pgs test 5", .nproc = 2, .arguments = ""},
            .alphaInit = 5.0,
            .domainLength = 2.0,
            .maxAlphaAllowed = {} /*default value*/,
            .maxDeltaPressureFac = {}, /*default value*/
            .dt = .1,
            .getFieldFunction =
                [](std::shared_ptr<ablate::eos::EOS> eos) {
                    auto flowState = std::make_shared<ablate::finiteVolume::fieldFunctions::CompressibleFlowState>(
                        eos, ablate::mathFunctions::Create(300.0), ablate::mathFunctions::Create("101325.0 + x*y*10000"), ablate::mathFunctions::Create("20.0*x, 10.0*y"));

                    return std::shared_ptr<ablate::mathFunctions::FieldFunction>{std::make_shared<ablate::finiteVolume::fieldFunctions::Euler>(flowState)};
                },
            /* expected results **/
            .expectedAlpha = 3.7061,
            .expectedMaxMach = 0.0633316},
        (PressureGradientScalingTestParameters){
            .mpiTestParameter = {.testName = "pgs test 6", .nproc = 2, .arguments = ""},
            .alphaInit = 5.0,
            .domainLength = 2.0,
            .maxAlphaAllowed = {} /*default value*/,
            .maxDeltaPressureFac = {}, /*default value*/
            .dt = .1,
            .getFieldFunction =
                [](std::shared_ptr<ablate::eos::EOS> eos) {
                    auto flowState = std::make_shared<ablate::finiteVolume::fieldFunctions::CompressibleFlowState>(
                        eos,
                        ablate::mathFunctions::Create(300.0),
                        std::make_shared<ablate::mathFunctions::geom::Sphere>(std::vector<double>{.75, .75}, .15, std::vector<double>{101325 * 1.5}, std::vector<double>{101325}),
                        std::make_shared<ablate::mathFunctions::geom::Sphere>(std::vector<double>{.75, .75}, .15, std::vector<double>{10, 10.0}, std::vector<double>{0.0, 0.0}));

                    return std::shared_ptr<ablate::mathFunctions::FieldFunction>{std::make_shared<ablate::finiteVolume::fieldFunctions::Euler>(flowState)};
                },
            /* expected results **/
            .expectedAlpha = 1.0,
            .expectedMaxMach = 0.0407333}),
    [](const testing::TestParamInfo<PressureGradientScalingTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });
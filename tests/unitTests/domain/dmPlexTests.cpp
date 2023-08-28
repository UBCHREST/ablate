#include <petsc.h>
#include <cmath>
#include <memory>
#include "domain/dmPlex.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "mpiTestFixture.hpp"
#include "petscTestErrorChecker.hpp"
#include "utilities/petscUtilities.hpp"

using namespace ablate;

struct DMPlexParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    // Optional parameters passed to dmPlex object (can be null)
    std::shared_ptr<parameters::Parameters> parameters;
};

class DMPlexTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<DMPlexParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(DMPlexTestFixture, ShouldCreateAndViewDMPlex) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();

            // Get the testing param
            auto &testingParam = GetParam();

            // act
            auto dmPlex = std::make_shared<ablate::domain::DMPlex>(
                std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>>{}, "dmPlex", std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{}, testingParam.parameters);

            // assert - print the dmPlex to standard out
            DMView(dmPlex->GetDM(), PETSC_VIEWER_STDOUT_WORLD) >> testErrorChecker;
        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(MeshTests, DMPlexTestFixture,
                         testing::Values((DMPlexParameters){.mpiTestParameter = testingResources::MpiTestParameter("default DMPlex", 1, "", "outputs/domain/dmPlex_NoArguments"),
                                                            .parameters = nullptr}),
                         [](const testing::TestParamInfo<DMPlexParameters> &info) { return info.param.mpiTestParameter.getTestName(); });

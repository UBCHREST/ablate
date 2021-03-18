static char help[] =
    "Time-dependent Low Mach Flow in 2d channels with finite elements.\n\
We solve the incompressible problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include <petsc.h>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"
#include "incompressibleFlow.h"
#include "mesh.h"

TEST(IncompressibleFlow, ShouldPackIncompressibleFlowParameters) {
    // arrange
    IncompressibleFlowParameters parameters{.strouhal = 1.0, .reynolds = 2.0, .peclet = 4.0, .mu = 7.0, .k = 8.0, .cp = 9.0};

    PetscScalar packedConstants[TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS];

    // act
    IncompressibleFlow_PackParameters(&parameters, packedConstants);

    // assert
    ASSERT_EQ(1.0, packedConstants[STROUHAL]) << " STROUHAL is incorrect";
    ASSERT_EQ(2.0, packedConstants[REYNOLDS]) << " REYNOLDS is incorrect";
    ASSERT_EQ(4.0, packedConstants[PECLET]) << " PECLET is incorrect";
    ASSERT_EQ(7.0, packedConstants[MU]) << " MU is incorrect";
    ASSERT_EQ(8.0, packedConstants[K]) << " K is incorrect";
    ASSERT_EQ(9.0, packedConstants[CP]) << " CP is incorrect";
}

class IncompressibleFlowParametersSetupTextFixture : public testingResources::MpiTestFixture,
                                                     public ::testing::WithParamInterface<std::tuple<testingResources::MpiTestParameter, IncompressibleFlowParameters>> {
   public:
    void SetUp() override { SetMpiParameters(std::get<0>(GetParam())); }
};

TEST_P(IncompressibleFlowParametersSetupTextFixture, ShouldParseFromPetscOptions) {
    StartWithMPI
        // arrange
        PetscErrorCode ierr = PetscInitialize(argc, argv, NULL, "");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        PetscBag petscFlowParametersBag;
        IncompressibleFlowParameters *actualParameters;

        // act
        IncompressibleFlow_ParametersFromPETScOptions(&petscFlowParametersBag);
        PetscBagGetData(petscFlowParametersBag, (void **)&actualParameters);

        // assert
        auto expectedParameters = std::get<1>(GetParam());
        ASSERT_EQ(actualParameters->strouhal, expectedParameters.strouhal) << " STROUHAL is incorrect";
        ASSERT_EQ(actualParameters->reynolds, expectedParameters.reynolds) << " REYNOLDS is incorrect";
        ASSERT_EQ(actualParameters->peclet, expectedParameters.peclet) << " PECLET is incorrect";
        ASSERT_EQ(actualParameters->mu, expectedParameters.mu) << " MU is incorrect";
        ASSERT_EQ(actualParameters->k, expectedParameters.k) << " K is incorrect";
        ASSERT_EQ(actualParameters->cp, expectedParameters.cp) << " CP is incorrect";

    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    IncompressibleFlow, IncompressibleFlowParametersSetupTextFixture,
    ::testing::Values(std::make_tuple(testingResources::MpiTestParameter{.testName = "default parameters", .nproc = 1, .arguments = ""},
                                      IncompressibleFlowParameters{.strouhal = 1.0, .reynolds = 1.0, .peclet = 1.0, .mu = 1.0, .k = 1.0, .cp = 1.0}),
                      std::make_tuple(testingResources::MpiTestParameter{.testName = "strouhal only", .nproc = 1, .arguments = "-strouhal 10.0"},
                                      IncompressibleFlowParameters{.strouhal = 10.0, .reynolds = 1.0, .peclet = 1.0, .mu = 1.0, .k = 1.0, .cp = 1.0}),
                      std::make_tuple(
                          testingResources::MpiTestParameter{
                              .testName = "all parameters",
                              .nproc = 1,
                              .arguments = "-strouhal 10.0 -reynolds 11.1 -froude 12.2 -peclet 13.3 -heatRelease 14.4 -gamma 15.5 -mu 16.6 -k 17.7 -cp 18.8 -beta 19.9 -gravityDirection 2 -pth 20.2"},
                          IncompressibleFlowParameters{.strouhal = 10.0, .reynolds = 11.1, .peclet = 13.3, .mu = 16.6, .k = 17.7, .cp = 18.8})),
    [](const testing::TestParamInfo<std::tuple<testingResources::MpiTestParameter, IncompressibleFlowParameters>> &info) { return std::get<0>(info.param).getTestName(); });
#include <petsc.h>
#include "gtest/gtest.h"
#include "parameters.h"
#include "testFixtures/MpiTestFixture.hpp"

TEST(ParametersTests, ShouldPackFlowParameters) {
    // arrange
    FlowParameters parameters{
        .strouhal = 1.0, .reynolds = 2.0, .froude = 3.0, .peclet = 4.0, .heatRelease = 5.0, .gamma = 6.0, .pth = 11, .mu = 7.0, .k = 8.0, .cp = 9.0, .beta = 10, .gravityDirection = 12};

    PetscScalar packedConstants[TOTAlCONSTANTS];

    // act
    PackFlowParameters(&parameters, packedConstants);

    // assert
    ASSERT_EQ(1.0, packedConstants[STROUHAL]) << " STROUHAL is incorrect";
    ASSERT_EQ(2.0, packedConstants[REYNOLDS]) << " REYNOLDS is incorrect";
    ASSERT_EQ(3.0, packedConstants[FROUDE]) << " FROUDE is incorrect";
    ASSERT_EQ(4.0, packedConstants[PECLET]) << " PECLET is incorrect";
    ASSERT_EQ(5.0, packedConstants[HEATRELEASE]) << " HEATRELEASE is incorrect";
    ASSERT_EQ(6.0, packedConstants[GAMMA]) << " GAMMA is incorrect";
    ASSERT_EQ(11.0, packedConstants[PTH]) << " PTH is incorrect";
    ASSERT_EQ(7.0, packedConstants[MU]) << " MU is incorrect";
    ASSERT_EQ(8.0, packedConstants[K]) << " K is incorrect";
    ASSERT_EQ(9.0, packedConstants[CP]) << " CP is incorrect";
    ASSERT_EQ(10.0, packedConstants[BETA]) << " BETA is incorrect";
    ASSERT_EQ(12.0, packedConstants[GRAVITY_DIRECTION]) << " BETA is incorrect";
}

class ParametersSetupTextFixture : public MpiTestFixture, public ::testing::WithParamInterface<std::tuple<MpiTestParameter, FlowParameters>> {
   public:
    void SetUp() override { SetMpiParameters(std::get<0>(GetParam())); }
};

TEST_P(ParametersSetupTextFixture, SetupAndParseConstants) {
    StartWithMPI
        // arrange
        PetscErrorCode ierr = PetscInitialize(argc, argv, NULL, "");
        CHKERRABORT(PETSC_COMM_WORLD, ierr);

        PetscBag petscFlowParametersBack;
        FlowParameters *actualParameters;

        // act
        SetupFlowParameters(&petscFlowParametersBack);
        PetscBagGetData(petscFlowParametersBack, (void **)&actualParameters);

        // assert
        auto expectedParameters = std::get<1>(GetParam());
        ASSERT_EQ(actualParameters->strouhal, expectedParameters.strouhal) << " STROUHAL is incorrect";
        ASSERT_EQ(actualParameters->reynolds, expectedParameters.reynolds) << " REYNOLDS is incorrect";
        ASSERT_EQ(actualParameters->froude, expectedParameters.froude) << " FROUDE is incorrect";
        ASSERT_EQ(actualParameters->peclet, expectedParameters.peclet) << " PECLET is incorrect";
        ASSERT_EQ(actualParameters->heatRelease, expectedParameters.heatRelease) << " HEATRELEASE is incorrect";
        ASSERT_EQ(actualParameters->gamma, expectedParameters.gamma) << " GAMMA is incorrect";
        ASSERT_EQ(actualParameters->pth, expectedParameters.pth) << " pth is incorrect";
        ASSERT_EQ(actualParameters->mu, expectedParameters.mu) << " MU is incorrect";
        ASSERT_EQ(actualParameters->k, expectedParameters.k) << " K is incorrect";
        ASSERT_EQ(actualParameters->cp, expectedParameters.cp) << " CP is incorrect";
        ASSERT_EQ(actualParameters->beta, expectedParameters.beta) << " BETA is incorrect";
        ASSERT_EQ(actualParameters->gravityDirection, expectedParameters.gravityDirection) << " gravityDirection is incorrect";
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    ParametersTests,
    ParametersSetupTextFixture,
    ::testing::Values(
        std::make_tuple(
            MpiTestParameter{.nproc = 1, .arguments = ""},
            FlowParameters{
                .strouhal = 1.0, .reynolds = 1.0, .froude = 1.0, .peclet = 1.0, .heatRelease = 1.0, .gamma = 1.0, .pth = 1, .mu = 1.0, .k = 1.0, .cp = 1.0, .beta = 1.0, .gravityDirection = 0}),
        std::make_tuple(
            MpiTestParameter{.nproc = 1, .arguments = "-strouhal 10.0"},
            FlowParameters{
                .strouhal = 10.0, .reynolds = 1.0, .froude = 1.0, .peclet = 1.0, .heatRelease = 1.0, .gamma = 1.0, .pth = 1, .mu = 1.0, .k = 1.0, .cp = 1.0, .beta = 1.0, .gravityDirection = 0}),
        std::make_tuple(
            MpiTestParameter{.nproc = 1,
                             .arguments = "-strouhal 10.0 -reynolds 11.1 -froude 12.2 -peclet 13.3 -heatRelease 14.4 -gamma 15.5 -mu 16.6 -k 17.7 -cp 18.8 -beta 19.9 -gravityDirection 2 -pth 20.2"},
            FlowParameters{.strouhal = 10.0,
                           .reynolds = 11.1,
                           .froude = 12.2,
                           .peclet = 13.3,
                           .heatRelease = 14.4,
                           .gamma = 15.5,
                           .pth = 20.2,
                           .mu = 16.6,
                           .k = 17.7,
                           .cp = 18.8,
                           .beta = 19.9,
                           .gravityDirection = 2})));
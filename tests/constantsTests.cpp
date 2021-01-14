#include <petsc.h>
#include "constants.h"
#include "gtest/gtest.h"
#include "testFixtures/MpiTestFixture.hpp"

TEST(ConstantsTests, ShouldPackFlowParameters) {
    // arrange
    FlowParameters parameters{.strouhal = 1.0, .reynolds = 2.0, .froude = 3.0, .peclet = 4.0, .heatRelease = 5.0, .gamma = 6.0, .mu = 7.0, .k = 8.0, .cp = 9.0, .beta = 10};

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
    ASSERT_EQ(7.0, packedConstants[MU]) << " MU is incorrect";
    ASSERT_EQ(8.0, packedConstants[K]) << " K is incorrect";
    ASSERT_EQ(9.0, packedConstants[CP]) << " CP is incorrect";
    ASSERT_EQ(10.0, packedConstants[BETA]) << " BETA is incorrect";
}

class ConstantsSetupTextFixture : public MpiTestFixture, public ::testing::WithParamInterface<std::tuple<MpiTestParameter, FlowParameters>> {
   public:
    void SetUp() override { SetMpiParameters(std::get<0>(GetParam())); }
};

TEST_P(ConstantsSetupTextFixture, SetupAndParseConstants) {
    // arrange

    // act

    // assert
}

INSTANTIATE_TEST_CASE_P(ConstantsTests,
                        ConstantsSetupTextFixture,
                        ::testing::Values(std::make_tuple(
                            MpiTestParameter{
                                .nproc = 1,
                                .expectedOutputFile = "outputs/2d_tri_p2_p1_p1_tconv",
                                .arguments = "-dm_plex_separate_marker -dm_refine 0 "
                                             "-vel_petscspace_degree 2 -pres_petscspace_degree 1 -temp_petscspace_degree 1 "
                                             "-ts_max_steps 4 -ts_dt 0.1 -ts_convergence_estimate -convest_num_refine 1 "
                                             "-snes_error_if_not_converged -snes_convergence_test correct_pressure "
                                             "-ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged "
                                             "-pc_type fieldsplit -pc_fieldsplit_0_fields 0,2 -pc_fieldsplit_1_fields 1 -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full "
                                             "-fieldsplit_0_pc_type lu "
                                             "-fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"}

                            ,
                            FlowParameters{.beta = 12})));
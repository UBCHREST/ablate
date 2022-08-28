#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <vector>
#include "finiteVolume/processes/les.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

typedef struct {
    PetscInt dim;
    PetscReal mut;
    std::vector<PetscReal> gradVel;
    std::vector<PetscReal> expectedlesStressTensor;
} StressTensorTestParameters;

class lesStressTensorTestFixture : public testing::TestWithParam<StressTensorTestParameters> {};

TEST_P(lesStressTensorTestFixture, ShouldComputeTheCorrectlesStressTensor) {
    // arrange
    PetscReal computedLESTau[9];
    const auto &params = GetParam();

    // act
    PetscErrorCode ierr = ablate::finiteVolume::processes::NavierStokesTransport::CompressibleFlowComputeStressTensor(params.dim, params.mut, params.gradVel.data(), computedLESTau);

    // assert
    ASSERT_EQ(0, ierr);
    for (auto c = 0; c < params.dim; c++) {
        for (auto d = 0; d < params.dim; d++) {
            auto i = c * params.dim + d;
            ASSERT_NEAR(computedLESTau[i], params.expectedlesStressTensor[i], 1E-8) << "The tau component [" + std::to_string(c) + "][" + std::to_string(d) + "] is incorrect";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(EulerTransportTests, lesStressTensorTestFixture,
                         testing::Values((StressTensorTestParameters){.dim = 1, .mut = .3, .gradVel = {3.5}, .expectedlesStressTensor = {1.4}},
                                         (StressTensorTestParameters){.dim = 2, .mut = .3, .gradVel = {3.5, -2.45, 0, 0.0}, .expectedlesStressTensor = {1.4, -0.735, -0.735, -0.7}},
                                         (StressTensorTestParameters){.dim = 2, .mut = 1.5, .gradVel = {3.5, -2.45, 0, -7}, .expectedlesStressTensor = {14, -3.675, -3.675, -17.5}},
                                         (StressTensorTestParameters){.dim = 2, .mut = 1.5, .gradVel = {0, -12, 12, 0}, .expectedlesStressTensor = {0, 0, 0, 0}},
                                         (StressTensorTestParameters){.dim = 3, .mut = 1.5, .gradVel = {1, 0, 0, 0, 2, 0, 0, 0, 3}, .expectedlesStressTensor = {-3, 0, 0, 0, 0, 0, 0, 0, 3}},
                                         (StressTensorTestParameters){.dim = 3, .mut = 1.5, .gradVel = {1, 2, 3, 4, 5, 6, 7, 8, 9}, .expectedlesStressTensor = {-12, 9, 15, 9, 0, 21, 15, 21, 12}},
                                         (StressTensorTestParameters){
                                             .dim = 3, .mut = 1.5, .gradVel = {-1, -2, -3, -4, -5, -6, -7, -8, -9}, .expectedlesStressTensor = {12, -9, -15, -9, 0, -21, -15, -21, -12}},
                                         (StressTensorTestParameters){.dim = 3, .mut = 0.0, .gradVel = {1, 2, 3, 4, 5, 6, 7, 8, 9}, .expectedlesStressTensor = {0, 0, 0, 0, 0, 0, 0, 0, 0}},
                                         (StressTensorTestParameters){.dim = 3, .mut = 0.7, .gradVel = {0, 0, 0, 0, 0, 0, 0, 0, 0}, .expectedlesStressTensor = {0, 0, 0, 0, 0, 0, 0, 0, 0}}),
                         [](const testing::TestParamInfo<StressTensorTestParameters> &info) { return "InputParameters_" + std::to_string(info.index); });

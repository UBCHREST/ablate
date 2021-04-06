#include <compressibleFlow.h>
#include <petsc.h>
#include <vector>
#include "asserts.hpp"
#include "gtest/gtest.h"

using namespace testingResources;

struct CompressibleFlowFluxTestParameters {
    std::string fluxDifferencer;
    std::vector<PetscReal> area;
    std::vector<PetscReal> xLeft;
    std::vector<PetscReal> xRight;
    PetscReal expectedRhoFlux;
    std::vector<PetscReal> expectedRhoUFlux;
    PetscReal expectedRhoEFlux;
};

class CompressibleFlowFluxTestFixture : public ::testing::TestWithParam<CompressibleFlowFluxTestParameters> {};

TEST_P(CompressibleFlowFluxTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto& params = GetParam();

    FlowData flowData;
    PetscErrorCode ierr = FlowCreate(&flowData);
    ASSERT_EQ(0, ierr);

    EulerFlowData* eulerFlowData;
    PetscNew(&eulerFlowData);
    eulerFlowData->cfl = NAN;
    eulerFlowData->gamma = 1.4;
    ierr = FluxDifferencerGet(params.fluxDifferencer.c_str(), &(eulerFlowData->fluxDifferencer));
    ASSERT_EQ(0, ierr);
    flowData->data = eulerFlowData;

    // act
    PetscReal computedRowFlux;
    CompressibleFlowComputeFluxRho(params.area.size(), 1, NULL, &params.area[0], &params.xLeft[0], &params.xRight[0], 0, NULL, &computedRowFlux, flowData);
    std::vector<PetscReal> computedRowUFlux(params.expectedRhoUFlux.size());
    CompressibleFlowComputeFluxRhoU(params.area.size(), 1, NULL, &params.area[0], &params.xLeft[0], &params.xRight[0], 0, NULL, &computedRowUFlux[0], flowData);
    PetscReal computedRowEFlux;
    CompressibleFlowComputeFluxRhoE(params.area.size(), 1, NULL, &params.area[0], &params.xLeft[0], &params.xRight[0], 0, NULL, &computedRowEFlux, flowData);

    // assert
    ASSERT_ABOUT(computedRowFlux, params.expectedRhoFlux, 1E-3);
    for (auto i = 0; i < params.expectedRhoUFlux.size(); i++) {
        ASSERT_ABOUT(computedRowUFlux[i], params.expectedRhoUFlux[i], 1E-3);
    }
    ASSERT_ABOUT(computedRowEFlux, params.expectedRhoEFlux, 1E-3);

    // cleanup
    ierr = FlowDestroy(&flowData);
    ASSERT_EQ(0, ierr);
}

INSTANTIATE_TEST_SUITE_P(FluxDifferencer, CompressibleFlowFluxTestFixture,
                         testing::Values((CompressibleFlowFluxTestParameters){.fluxDifferencer = "ausm",
                                                                              .area = {1},
                                                                              .xLeft = {0.400688, 0.371908, 0.929113},
                                                                              .xRight = {0.391646, 0.363631, 0.924943},
                                                                              .expectedRhoFlux = 0.371703,
                                                                              .expectedRhoUFlux = {0.648038, 0},
                                                                              .expectedRhoEFlux = 1.142619},
                                         (CompressibleFlowFluxTestParameters){.fluxDifferencer = "ausm",
                                                                              .area = {1},
                                                                              .xLeft = {5.999240, 117.570106, 2304.275075},
                                                                              .xRight = {5.992420, -37.131012, 230.275501},
                                                                              .expectedRhoFlux = 0.091875,
                                                                              .expectedRhoUFlux = {508.789524, 0},
                                                                              .expectedRhoEFlux = 42.347103},
                                         (CompressibleFlowFluxTestParameters){.fluxDifferencer = "ausm",
                                                                              .area = {1},
                                                                              .xLeft = {0.864333, 1.637664, 2.369795},
                                                                              .xRight = {0.893851, 1.714786, 2.501471},
                                                                              .expectedRhoFlux = 1.637664,
                                                                              .expectedRhoUFlux = {3.430243, 0},
                                                                              .expectedRhoEFlux = 5.110295},
                                         (CompressibleFlowFluxTestParameters){.fluxDifferencer = "ausm",
                                                                              .area = {-1.0},
                                                                              .xLeft = {0.893851, 1.714786, 2.501471},
                                                                              .xRight = {0.864333, 1.637664, 2.369795},
                                                                              .expectedRhoFlux = 1.637664,
                                                                              .expectedRhoUFlux = {-3.430243, 0},
                                                                              .expectedRhoEFlux = 5.110295}
                                         ));

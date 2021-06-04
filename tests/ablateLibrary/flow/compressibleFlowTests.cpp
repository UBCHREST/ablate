#include <compressibleFlow.h>
#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <vector>
#include "gtest/gtest.h"

struct CompressibleFlowFluxTestParameters {
    std::string fluxDifferencer;
    std::vector<PetscReal> area;
    std::vector<PetscReal> xLeft;
    std::vector<PetscReal> xRight;
    std::vector<PetscReal> expectedFlux;
};

class CompressibleFlowFluxTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<CompressibleFlowFluxTestParameters> {};

TEST_P(CompressibleFlowFluxTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto& params = GetParam();

    FlowData_CompressibleFlow eulerFlowData;
    PetscNew(&eulerFlowData);
    eulerFlowData->cfl = NAN;

    FluxDifferencerGet(params.fluxDifferencer.c_str(), &(eulerFlowData->fluxDifferencer)) >> errorChecker;

    // set a perfect gas for testing
    EOSData eos;
    EOSCreate(&eos);
    EOSSetType(eos, "perfectGas");
    EOSSetFromOptions(eos);
    eulerFlowData->eos = eos;

    // act
    std::vector<PetscReal> computedFlux(params.expectedFlux.size());
    CompressibleFlowComputeEulerFlux(params.area.size(), 1, NULL, &params.area[0], &params.xLeft[0], &params.xRight[0], 0, NULL, &computedFlux[0], eulerFlowData);

    // assert
    for (auto i = 0; i < params.expectedFlux.size(); i++) {
        ASSERT_NEAR(computedFlux[i], params.expectedFlux[i], 1E-3);
    }

    // cleanup
    EOSDestroy(&eos) >> errorChecker;
}

INSTANTIATE_TEST_SUITE_P(
    CompressibleFlow, CompressibleFlowFluxTestFixture,
    testing::Values(
        (CompressibleFlowFluxTestParameters){
            .fluxDifferencer = "ausm", .area = {1}, .xLeft = {0.400688, 0.929113, 0.371908}, .xRight = {0.391646, 0.924943, 0.363631}, .expectedFlux = {0.371703, 1.142619, 0.648038}},
        (CompressibleFlowFluxTestParameters){
            .fluxDifferencer = "ausm", .area = {1}, .xLeft = {5.999240, 2304.275075, 117.570106}, .xRight = {5.992420, 230.275501, -37.131012}, .expectedFlux = {0.091875, 42.347103, 508.789524}},
        (CompressibleFlowFluxTestParameters){.fluxDifferencer = "ausm",
                                             .area = {1},
                                             .xLeft =
                                                 {
                                                     0.864333,
                                                     2.369795,
                                                     1.637664,
                                                 },
                                             .xRight = {0.893851, 2.501471, 1.714786},
                                             .expectedFlux = {1.637664, 5.110295, 3.430243}},
        (CompressibleFlowFluxTestParameters){
            .fluxDifferencer = "ausm", .area = {-1.0}, .xLeft = {0.893851, 2.501471, 1.714786}, .xRight = {0.864333, 2.369795, 1.637664}, .expectedFlux = {-1.637664, -5.110295, -3.430243}}));

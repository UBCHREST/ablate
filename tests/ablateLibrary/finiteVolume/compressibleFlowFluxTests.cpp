#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <finiteVolume/processes/eulerAdvection.hpp>
#include <vector>
#include "eos/perfectGas.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

struct CompressibleFlowFluxTestParameters {
    std::shared_ptr<ablate::finiteVolume::fluxCalculator::FluxCalculator> fluxCalculator;
    std::vector<PetscReal> area;
    std::vector<PetscReal> xLeft;
    std::vector<PetscReal> xRight;
    std::vector<PetscReal> expectedFlux;
};

class CompressibleFlowFluxTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<CompressibleFlowFluxTestParameters> {};

TEST_P(CompressibleFlowFluxTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto& params = GetParam();

    // For this test, manually setup the compressible flow object;
    ablate::finiteVolume::processes::EulerAdvection::EulerAdvectionData eulerFlowData;
    PetscNew(&eulerFlowData);
    eulerFlowData->cfl = NAN;
    eulerFlowData->fluxCalculatorFunction = params.fluxCalculator->GetFluxCalculatorFunction();

    // set a perfect gas for testing
    auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>());
    eulerFlowData->decodeStateFunction = eos->GetDecodeStateFunction();
    eulerFlowData->decodeStateFunctionContext = eos->GetDecodeStateContext();

    // setup a fake PetscFVFaceGeom
    PetscFVFaceGeom faceGeom{};
    std::copy(std::begin(params.area), std::end(params.area), faceGeom.normal);

    // act
    std::vector<PetscReal> computedFlux(params.expectedFlux.size());
    PetscInt uOff[1] = {0};
    /*CompressibleFlowComputeEulerFlux ( PetscInt dim, const PetscFVFaceGeom* fg, const PetscFVCellGeom* cgL, const PetscFVCellGeom* cgR,
            const PetscInt uOff[], const PetscScalar uL[], const PetscScalar uR[], const PetscScalar* gradL[], const PetscScalar* gradR[],
            const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar* gradAuxL[], const PetscScalar* gradAuxR[],
            PetscScalar* flux, void* ctx)*/
    ablate::finiteVolume::processes::EulerAdvection::CompressibleFlowComputeEulerFlux(
        params.area.size(), &faceGeom, uOff, NULL, &params.xLeft[0], &params.xRight[0], NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &computedFlux[0], eulerFlowData);

    // assert
    for (std::size_t i = 0; i < params.expectedFlux.size(); i++) {
        ASSERT_NEAR(computedFlux[i], params.expectedFlux[i], 1E-3);
    }

    // cleanup
    PetscFree(eulerFlowData);
}

INSTANTIATE_TEST_SUITE_P(CompressibleFlow, CompressibleFlowFluxTestFixture,
                         testing::Values((CompressibleFlowFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                              .area = {1},
                                                                              .xLeft = {0.400688, 0.929113, 0.371908},
                                                                              .xRight = {0.391646, 0.924943, 0.363631},
                                                                              .expectedFlux = {0.371703, 1.142619, 0.648038}},
                                         (CompressibleFlowFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                              .area = {1},
                                                                              .xLeft = {5.999240, 2304.275075, 117.570106},
                                                                              .xRight = {5.992420, 230.275501, -37.131012},
                                                                              .expectedFlux = {0.091875, 42.347103, 508.789524}},
                                         (CompressibleFlowFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                              .area = {1},
                                                                              .xLeft =
                                                                                  {
                                                                                      0.864333,
                                                                                      2.369795,
                                                                                      1.637664,
                                                                                  },
                                                                              .xRight = {0.893851, 2.501471, 1.714786},
                                                                              .expectedFlux = {1.637664, 5.110295, 3.430243}},
                                         (CompressibleFlowFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                              .area = {-1.0},
                                                                              .xLeft = {0.893851, 2.501471, 1.714786},
                                                                              .xRight = {0.864333, 2.369795, 1.637664},
                                                                              .expectedFlux = {-1.637664, -5.110295, -3.430243}}));

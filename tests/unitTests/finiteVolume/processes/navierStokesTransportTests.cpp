#include <petsc.h>
#include <petscTestFixture.hpp>
#include <vector>
#include "domain/mockField.hpp"
#include "eos/perfectGas.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

struct NavierStokesTransportFluxTestParameters {
    std::shared_ptr<ablate::finiteVolume::fluxCalculator::FluxCalculator> fluxCalculator;
    std::vector<PetscReal> area;
    std::vector<PetscReal> xLeft;
    std::vector<PetscReal> xRight;
    std::vector<PetscReal> expectedFlux;
};

class NavierStokesTransportFluxTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<NavierStokesTransportFluxTestParameters> {};

TEST_P(NavierStokesTransportFluxTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto &params = GetParam();

    // For this test, manually setup the compressible flow object;
    ablate::finiteVolume::processes::NavierStokesTransport::AdvectionData eulerFlowData;
    eulerFlowData.cfl = NAN;
    eulerFlowData.fluxCalculatorFunction = params.fluxCalculator->GetFluxCalculatorFunction();

    // set a perfect gas for testing
    auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>());
    auto eulerFieldMock = ablateTesting::domain::MockField::Create("euler", 3);
    eulerFlowData.computeTemperature = eos->GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::Temperature, {eulerFieldMock});
    eulerFlowData.computeInternalEnergy = eos->GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {eulerFieldMock});
    eulerFlowData.computeSpeedOfSound = eos->GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SpeedOfSound, {eulerFieldMock});
    eulerFlowData.computePressure = eos->GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::Pressure, {eulerFieldMock});

    // setup a fake PetscFVFaceGeom
    PetscFVFaceGeom faceGeom{};
    std::copy(std::begin(params.area), std::end(params.area), faceGeom.normal);

    // act
    std::vector<PetscReal> computedFlux(params.expectedFlux.size());
    PetscInt uOff[1] = {0};
    PetscInt aOff[1] = {0};
    PetscReal TempGuess[1] = {300};
    ablate::finiteVolume::processes::NavierStokesTransport::AdvectionFlux(
        params.area.size(), &faceGeom, uOff, &params.xLeft[0], &params.xRight[0], aOff, TempGuess, TempGuess, &computedFlux[0], &eulerFlowData);

    // assert
    for (std::size_t i = 0; i < params.expectedFlux.size(); i++) {
        ASSERT_NEAR(computedFlux[i], params.expectedFlux[i], 1E-3);
    }
}

INSTANTIATE_TEST_SUITE_P(EulerTransportTests, NavierStokesTransportFluxTestFixture,
                         testing::Values((NavierStokesTransportFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                                   .area = {1},
                                                                                   .xLeft = {0.400688, 0.929113, 0.371908},
                                                                                   .xRight = {0.391646, 0.924943, 0.363631},
                                                                                   .expectedFlux = {0.371703, 1.142619, 0.648038}},
                                         (NavierStokesTransportFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                                   .area = {1},
                                                                                   .xLeft = {5.999240, 2304.275075, 117.570106},
                                                                                   .xRight = {5.992420, 230.275501, -37.131012},
                                                                                   .expectedFlux = {0.091875, 42.347103, 508.789524}},
                                         (NavierStokesTransportFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                                   .area = {1},
                                                                                   .xLeft =
                                                                                       {
                                                                                           0.864333,
                                                                                           2.369795,
                                                                                           1.637664,
                                                                                       },
                                                                                   .xRight = {0.893851, 2.501471, 1.714786},
                                                                                   .expectedFlux = {1.637664, 5.110295, 3.430243}},
                                         (NavierStokesTransportFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                                   .area = {-1.0},
                                                                                   .xLeft = {0.893851, 2.501471, 1.714786},
                                                                                   .xRight = {0.864333, 2.369795, 1.637664},
                                                                                   .expectedFlux = {-1.637664, -5.110295, -3.430243}}));

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
    PetscInt dim;
    PetscReal mu;
    std::vector<PetscReal> gradVel;
    std::vector<PetscReal> expectedStressTensor;
} StressTensorTestParameters;

class StressTensorTestFixture : public testing::TestWithParam<StressTensorTestParameters> {};

TEST_P(StressTensorTestFixture, ShouldComputeTheCorrectStressTensor) {
    // arrange
    PetscReal computedTau[9];
    const auto &params = GetParam();

    // act //assert
    ASSERT_EQ(0, ablate::finiteVolume::processes::NavierStokesTransport::CompressibleFlowComputeStressTensor(params.dim, params.mu, params.gradVel.data(), computedTau));

    for (auto c = 0; c < params.dim; c++) {
        for (auto d = 0; d < params.dim; d++) {
            auto i = c * params.dim + d;
            ASSERT_NEAR(computedTau[i], params.expectedStressTensor[i], 1E-8) << "The tau component [" + std::to_string(c) + "][" + std::to_string(d) + "] is incorrect";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(EulerTransportTests, StressTensorTestFixture,
                         testing::Values((StressTensorTestParameters){.dim = 1, .mu = .3, .gradVel = {3.5}, .expectedStressTensor = {1.4}},
                                         (StressTensorTestParameters){.dim = 2, .mu = .3, .gradVel = {3.5, -2.45, 0, 0.0}, .expectedStressTensor = {1.4, -0.735, -0.735, -0.7}},
                                         (StressTensorTestParameters){.dim = 2, .mu = 1.5, .gradVel = {3.5, -2.45, 0, -7}, .expectedStressTensor = {14, -3.675, -3.675, -17.5}},
                                         (StressTensorTestParameters){.dim = 2, .mu = 1.5, .gradVel = {0, -12, 12, 0}, .expectedStressTensor = {0, 0, 0, 0}},
                                         (StressTensorTestParameters){.dim = 3, .mu = 1.5, .gradVel = {1, 0, 0, 0, 2, 0, 0, 0, 3}, .expectedStressTensor = {-3, 0, 0, 0, 0, 0, 0, 0, 3}},
                                         (StressTensorTestParameters){.dim = 3, .mu = 1.5, .gradVel = {1, 2, 3, 4, 5, 6, 7, 8, 9}, .expectedStressTensor = {-12, 9, 15, 9, 0, 21, 15, 21, 12}},
                                         (StressTensorTestParameters){
                                             .dim = 3, .mu = 1.5, .gradVel = {-1, -2, -3, -4, -5, -6, -7, -8, -9}, .expectedStressTensor = {12, -9, -15, -9, 0, -21, -15, -21, -12}},
                                         (StressTensorTestParameters){.dim = 3, .mu = 0.0, .gradVel = {1, 2, 3, 4, 5, 6, 7, 8, 9}, .expectedStressTensor = {0, 0, 0, 0, 0, 0, 0, 0, 0}},
                                         (StressTensorTestParameters){.dim = 3, .mu = 0.7, .gradVel = {0, 0, 0, 0, 0, 0, 0, 0, 0}, .expectedStressTensor = {0, 0, 0, 0, 0, 0, 0, 0, 0}}),
                         [](const testing::TestParamInfo<StressTensorTestParameters> &info) { return "InputParameters_" + std::to_string(info.index); });
#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <finiteVolume/processes/eulerTransport.hpp>
#include <vector>
#include "eos/perfectGas.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

struct EulerTransportFluxTestParameters {
    std::shared_ptr<ablate::finiteVolume::fluxCalculator::FluxCalculator> fluxCalculator;
    std::vector<PetscReal> area;
    std::vector<PetscReal> xLeft;
    std::vector<PetscReal> xRight;
    std::vector<PetscReal> expectedFlux;
};

class EulerTransportFluxTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EulerTransportFluxTestParameters> {};

TEST_P(EulerTransportFluxTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto &params = GetParam();

    // For this test, manually setup the compressible flow object;
    ablate::finiteVolume::processes::EulerTransport::AdvectionData eulerFlowData;
    eulerFlowData.cfl = NAN;
    eulerFlowData.fluxCalculatorFunction = params.fluxCalculator->GetFluxCalculatorFunction();

    // set a perfect gas for testing
    auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>());
    auto eulerFieldMock = ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 0};
    eulerFlowData.computeTemperature = eos->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, {eulerFieldMock});
    eulerFlowData.computeInternalEnergy = eos->GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {eulerFieldMock});
    eulerFlowData.computeSpeedOfSound = eos->GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SpeedOfSound, {eulerFieldMock});
    eulerFlowData.computePressure = eos->GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::Pressure, {eulerFieldMock});

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
    ablate::finiteVolume::processes::EulerTransport::AdvectionFlux(
        params.area.size(), &faceGeom, uOff, NULL, &params.xLeft[0], &params.xRight[0], NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &computedFlux[0], &eulerFlowData);

    // assert
    for (std::size_t i = 0; i < params.expectedFlux.size(); i++) {
        ASSERT_NEAR(computedFlux[i], params.expectedFlux[i], 1E-3);
    }
}

INSTANTIATE_TEST_SUITE_P(EulerTransportTests, EulerTransportFluxTestFixture,
                         testing::Values((EulerTransportFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                            .area = {1},
                                                                            .xLeft = {0.400688, 0.929113, 0.371908},
                                                                            .xRight = {0.391646, 0.924943, 0.363631},
                                                                            .expectedFlux = {0.371703, 1.142619, 0.648038}},
                                         (EulerTransportFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                            .area = {1},
                                                                            .xLeft = {5.999240, 2304.275075, 117.570106},
                                                                            .xRight = {5.992420, 230.275501, -37.131012},
                                                                            .expectedFlux = {0.091875, 42.347103, 508.789524}},
                                         (EulerTransportFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                            .area = {1},
                                                                            .xLeft =
                                                                                {
                                                                                    0.864333,
                                                                                    2.369795,
                                                                                    1.637664,
                                                                                },
                                                                            .xRight = {0.893851, 2.501471, 1.714786},
                                                                            .expectedFlux = {1.637664, 5.110295, 3.430243}},
                                         (EulerTransportFluxTestParameters){.fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                            .area = {-1.0},
                                                                            .xLeft = {0.893851, 2.501471, 1.714786},
                                                                            .xRight = {0.864333, 2.369795, 1.637664},
                                                                            .expectedFlux = {-1.637664, -5.110295, -3.430243}}));

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
    PetscInt dim;
    PetscReal mu;
    std::vector<PetscReal> gradVelL;
    std::vector<PetscReal> gradVelR;
    std::vector<PetscReal> expectedStressTensor;
} StressTensorTestParameters;

class StressTensorTestFixture : public testing::TestWithParam<StressTensorTestParameters> {};

TEST_P(StressTensorTestFixture, ShouldComputeTheCorrectStressTensor) {
    // arrange
    PetscReal computedTau[9];
    const auto &params = GetParam();

    // act
//    PetscErrorCode ierr = ablate::finiteVolume::processes::EulerTransport::CompressibleFlowComputeStressTensor(params.dim, params.mu, &params.gradVelL[0], &params.gradVelR[0], computedTau);

    // assert
//    ASSERT_EQ(0, ierr);
    for (auto c = 0; c < params.dim; c++) {
        for (auto d = 0; d < params.dim; d++) {
            auto i = c * params.dim + d;
            ASSERT_NEAR(computedTau[i], params.expectedStressTensor[i], 1E-8) << "The tau component [" + std::to_string(c) + "][" + std::to_string(d) + "] is incorrect";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    EulerTransportTests, StressTensorTestFixture,
    testing::Values(
        (StressTensorTestParameters){.dim = 1, .mu = .3, .gradVelL = {3.5}, .gradVelR = {3.5}, .expectedStressTensor = {1.4}},
        (StressTensorTestParameters){.dim = 1, .mu = .3, .gradVelL = {4.5}, .gradVelR = {2.5}, .expectedStressTensor = {1.4}},
        (StressTensorTestParameters){.dim = 2, .mu = .3, .gradVelL = {3.5, -2.45, 0, -1}, .gradVelR = {3.5, -2.45, 0, 1}, .expectedStressTensor = {1.4, -0.735, -0.735, -0.7}},
        (StressTensorTestParameters){.dim = 2, .mu = 1.5, .gradVelL = {3.5, -2.45, 0, -6}, .gradVelR = {3.5, -2.45, 0, -8}, .expectedStressTensor = {14, -3.675, -3.675, -17.5}},
        (StressTensorTestParameters){.dim = 2, .mu = 1.5, .gradVelL = {0, -12, 12, 0}, .gradVelR = {0, -12, 12, 0}, .expectedStressTensor = {0, 0, 0, 0}},
        (StressTensorTestParameters){.dim = 2, .mu = 1.5, .gradVelL = {0, -10, 12, 0}, .gradVelR = {0, -20, 12, 0}, .expectedStressTensor = {0, -4.5, -4.5, 0}},
        (StressTensorTestParameters){.dim = 3, .mu = 1.5, .gradVelL = {1, 0, 0, 0, 1, 0, 0, 0, 1}, .gradVelR = {1, 0, 0, 0, 3, 0, 0, 0, 5}, .expectedStressTensor = {-3, 0, 0, 0, 0, 0, 0, 0, 3}},
        (StressTensorTestParameters){
            .dim = 3, .mu = 1.5, .gradVelL = {2, 4, 6, 8, 10, 12, 14, 16, 18}, .gradVelR = {0, 0, 0, 0, 0, 0, 0, 0, 0}, .expectedStressTensor = {-12, 9, 15, 9, 0, 21, 15, 21, 12}},
        (StressTensorTestParameters){
            .dim = 3, .mu = 1.5, .gradVelL = {0, 0, 0, 0, 0, 0, 0, 0, 0}, .gradVelR = {-2, -4, -6, -8, -10, -12, -14, -16, -18}, .expectedStressTensor = {12, -9, -15, -9, 0, -21, -15, -21, -12}},
        (StressTensorTestParameters){
            .dim = 3, .mu = 1.5, .gradVelL = {2, 4, 6, 8, 10, 12, 14, 16, 18}, .gradVelR = {-2, -4, -6, -8, -10, -12, -14, -16, -18}, .expectedStressTensor = {0, 0, 0, 0, 0, 0, 0, 0, 0}},
        (StressTensorTestParameters){.dim = 3, .mu = 0.0, .gradVelL = {1, 2, 3, 4, 5, 6, 7, 8, 9}, .gradVelR = {1, 2, 3, 4, 5, 6, 7, 8, 9}, .expectedStressTensor = {0, 0, 0, 0, 0, 0, 0, 0, 0}},
        (StressTensorTestParameters){.dim = 3, .mu = 0.7, .gradVelL = {0, 0, 0, 0, 0, 0, 0, 0, 0}, .gradVelR = {0, 0, 0, 0, 0, 0, 0, 0, 0}, .expectedStressTensor = {0, 0, 0, 0, 0, 0, 0, 0, 0}}),
    [](const testing::TestParamInfo<StressTensorTestParameters> &info) { return "InputParameters_" + std::to_string(info.index); });
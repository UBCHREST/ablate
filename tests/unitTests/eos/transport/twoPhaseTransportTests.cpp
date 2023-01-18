#include "PetscTestFixture.hpp"
#include "eos/perfectGas.hpp"
#include "eos/transport/constant.hpp"
#include "eos/transport/sutherland.hpp"
#include "eos/transport/twoPhaseTransport.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

struct TwoPhaseTransportTestParameters {
    std::shared_ptr<ablate::eos::transport::TransportModel> transport1;
    std::shared_ptr<ablate::eos::transport::TransportModel> transport2;
    const PetscReal conservedIn[1];

    PetscReal expectedConductivity;
    PetscReal expectedViscosity;
    PetscReal expectedDiffusivity;
};

class TwoPhaseTransportTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TwoPhaseTransportTestParameters> {};

TEST_P(TwoPhaseTransportTestFixture, ShouldComputeCorrectDirectConductivity) {
    // ARRANGE
    auto transport1 = GetParam().transport1;
    auto transport2 = GetParam().transport2;

    auto twoPhaseModel = std::make_shared<ablate::eos::transport::TwoPhaseTransport>(transport1, transport2);
    auto conductivityFunction = twoPhaseModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});

    // ACT
    PetscReal computedk = NAN;
    conductivityFunction.function(GetParam().conservedIn, &computedk, conductivityFunction.context.get());

    // ASSERT
    double error = PetscAbs((GetParam().expectedConductivity - computedk) / GetParam().expectedConductivity);
    ASSERT_LT(error, 1E-5);
}

TEST_P(TwoPhaseTransportTestFixture, ShouldComputeCorrectDirectViscosity) {
    // ARRANGE
    auto transport1 = GetParam().transport1;
    auto transport2 = GetParam().transport2;

    auto twoPhaseModel = std::make_shared<ablate::eos::transport::TwoPhaseTransport>(transport1, transport2);
    auto viscosityFunction = twoPhaseModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});

    // ACT
    PetscReal computedmu = NAN;
    viscosityFunction.function(GetParam().conservedIn, &computedmu, viscosityFunction.context.get());

    // ASSERT
    double error = PetscAbs((GetParam().expectedViscosity - computedmu) / GetParam().expectedViscosity);
    ASSERT_LT(error, 1E-5);
}

TEST_P(TwoPhaseTransportTestFixture, ShouldComputeCorrectDirectDiffusivity) {
    // ARRANGE
    auto transport1 = GetParam().transport1;
    auto transport2 = GetParam().transport2;

    auto twoPhaseModel = std::make_shared<ablate::eos::transport::TwoPhaseTransport>(transport1, transport2);
    auto diffusivityFunction = twoPhaseModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, {});

    // ACT
    PetscReal computeddiff = NAN;
    diffusivityFunction.function(GetParam().conservedIn, &computeddiff, diffusivityFunction.context.get());

    // ASSERT
    double error = PetscAbs(GetParam().expectedDiffusivity - computeddiff);  // cannot divide by expected, would be dividing by zero
    ASSERT_LT(error, 1E-5);
}

INSTANTIATE_TEST_SUITE_P(TwoPhaseTransportTests, TwoPhaseTransportTestFixture,
                         testing::Values((TwoPhaseTransportTestParameters){.transport1 = std::make_shared<ablate::eos::transport::Constant>(double{1.0}, double{2.0}),
                                                                           .transport2 = std::make_shared<ablate::eos::transport::Constant>(double{3.0}, double{4.0}),
                                                                           .conservedIn = {0.75},
                                                                           .expectedConductivity = 1.5,
                                                                           .expectedViscosity = 2.5,
                                                                           .expectedDiffusivity = 0.0}),
                         [](const testing::TestParamInfo<TwoPhaseTransportTestParameters>& info) { return std::to_string(info.index); });
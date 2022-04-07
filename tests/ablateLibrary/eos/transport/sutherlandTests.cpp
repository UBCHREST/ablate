#include <eos/mockEOS.hpp>
#include <vector>
#include "eos/transport/sutherland.hpp"
#include "gtest/gtest.h"

struct SutherlandTransportTestParameters {
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal cpIn;

    PetscReal expectedConductivity;
    PetscReal expectedViscosity;
    PetscReal expectedDiffusivity;
};

class SutherlandTransportTestFixture : public ::testing::TestWithParam<SutherlandTransportTestParameters> {};

TEST_P(SutherlandTransportTestFixture, ShouldComputeCorrectDirectConductivity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = GetParam().temperatureIn; })));
    EXPECT_CALL(*eos, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(
            ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction([](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = GetParam().cpIn; })));

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);
    auto conductivityFunction = sutherlandModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});

    // ACT
    PetscReal computedK = NAN;
    conductivityFunction.function(nullptr, &computedK, conductivityFunction.context.get());

    // ASSERT
    double error = PetscAbs((GetParam().expectedConductivity - computedK) / GetParam().expectedConductivity);
    ASSERT_LT(error, 1E-5);
}

TEST_P(SutherlandTransportTestFixture, ShouldComputeCorrectTemperatureConductivity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(
            ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction([](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = GetParam().cpIn; })));

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);
    auto conductivityFunction = sutherlandModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, {});

    // ACT
    PetscReal computedK = NAN;
    conductivityFunction.function(nullptr, GetParam().temperatureIn, &computedK, conductivityFunction.context.get());

    // ASSERT
    double error = PetscAbs((GetParam().expectedConductivity - computedK) / GetParam().expectedConductivity);
    ASSERT_LT(error, 1E-5);
}

TEST_P(SutherlandTransportTestFixture, ShouldComputeDirectCorrectViscosity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = GetParam().temperatureIn; })));

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);
    auto viscosityFunction = sutherlandModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});

    // ACT
    PetscReal computedMu = NAN;
    viscosityFunction.function(nullptr, &computedMu, viscosityFunction.context.get());

    // ASSERT
    double error = PetscAbs((GetParam().expectedViscosity - computedMu) / GetParam().expectedViscosity);
    ASSERT_LT(error, 1E-5);
}

TEST_P(SutherlandTransportTestFixture, ShouldComputeTemperatureCorrectViscosity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);
    auto viscosityFunction = sutherlandModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, {});

    // ACT
    PetscReal computedMu = NAN;
    viscosityFunction.function(nullptr, GetParam().temperatureIn, &computedMu, viscosityFunction.context.get());

    // ASSERT
    double error = PetscAbs((GetParam().expectedViscosity - computedMu) / GetParam().expectedViscosity);
    ASSERT_LT(error, 1E-5);
}

TEST_P(SutherlandTransportTestFixture, ShouldComputeDirectCorrectDiffusivity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = GetParam().temperatureIn; })));
    EXPECT_CALL(*eos, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Density, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = GetParam().densityIn; })));

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);
    auto diffusivityFunction = sutherlandModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, {});

    // ACT
    PetscReal computedDiff = NAN;
    diffusivityFunction.function(nullptr, &computedDiff, diffusivityFunction.context.get());

    // ASSERT
    double error = PetscAbs((GetParam().expectedDiffusivity - computedDiff) / GetParam().expectedDiffusivity);
    ASSERT_LT(error, 1E-5);
}

TEST_P(SutherlandTransportTestFixture, ShouldComputeTemperatureCorrectDiffusivity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Density, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = GetParam().densityIn; })));

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);
    auto diffusivityFunction = sutherlandModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, {});

    // ACT
    PetscReal computedDiff = NAN;
    diffusivityFunction.function(nullptr, GetParam().temperatureIn, &computedDiff, diffusivityFunction.context.get());

    // ASSERT
    double error = PetscAbs((GetParam().expectedDiffusivity - computedDiff) / GetParam().expectedDiffusivity);
    ASSERT_LT(error, 1E-5);
}

INSTANTIATE_TEST_SUITE_P(SutherlandTests, SutherlandTransportTestFixture,
                         testing::Values((SutherlandTransportTestParameters){
                             .temperatureIn = 300.0, .densityIn = 1.1, .cpIn = 1001.1, .expectedConductivity = 0.02615186, .expectedViscosity = 1.8469051E-5, .expectedDiffusivity = 2.374829E-5}));

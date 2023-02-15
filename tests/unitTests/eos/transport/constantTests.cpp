#include "eos/transport/constant.hpp"
#include "gtest/gtest.h"

TEST(ConstantTransportTests, ShouldRecordConstantValuesForDirectFunction) {
    // ARRANGE
    const PetscReal expectedK = .234;
    const PetscReal expectedMu = 10.213;
    const PetscReal expectedDiff = 1.32;

    auto constantModel = std::make_shared<ablate::eos::transport::Constant>(expectedK, expectedMu, expectedDiff);

    auto conductivityFunction = constantModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto viscosityFunction = constantModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    auto diffusivityFunction = constantModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, {});

    // ACT
    PetscReal computedK = NAN;
    PetscReal computedMu = NAN;
    PetscReal computedDiff = NAN;

    conductivityFunction.function(nullptr, &computedK, conductivityFunction.context.get());
    viscosityFunction.function(nullptr, &computedMu, viscosityFunction.context.get());
    diffusivityFunction.function(nullptr, &computedDiff, diffusivityFunction.context.get());

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedK, computedK);
    ASSERT_DOUBLE_EQ(computedMu, computedMu);
    ASSERT_DOUBLE_EQ(computedDiff, computedDiff);
}

TEST(ConstantTransportTests, ShouldReturnNullFunctionsIfAllValuesZeroForDirectFunction) {
    // ARRANGE
    auto constantModel = std::make_shared<ablate::eos::transport::Constant>();

    // ACT
    // ASSERT
    ASSERT_TRUE(constantModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, {}).function == nullptr);
    ASSERT_TRUE(constantModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, {}).function == nullptr);
    ASSERT_TRUE(constantModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, {}).function == nullptr);
}

TEST(ConstantTransportTests, ShouldRecordConstantValuesForTemperatureFunction) {
    // ARRANGE
    const PetscReal expectedK = .234;
    const PetscReal expectedMu = 10.213;
    const PetscReal expectedDiff = 1.32;

    auto constantModel = std::make_shared<ablate::eos::transport::Constant>(expectedK, expectedMu, expectedDiff);

    auto conductivityFunction = constantModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, {});
    auto viscosityFunction = constantModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, {});
    auto diffusivityFunction = constantModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, {});

    // ACT
    PetscReal computedK = NAN;
    PetscReal computedMu = NAN;
    PetscReal computedDiff = NAN;

    conductivityFunction.function(nullptr, NAN, &computedK, conductivityFunction.context.get());
    viscosityFunction.function(nullptr, NAN, &computedMu, viscosityFunction.context.get());
    diffusivityFunction.function(nullptr, NAN, &computedDiff, diffusivityFunction.context.get());

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedK, computedK);
    ASSERT_DOUBLE_EQ(computedMu, computedMu);
    ASSERT_DOUBLE_EQ(computedDiff, computedDiff);
    ASSERT_DOUBLE_EQ(conductivityFunction.propertySize, 1);
    ASSERT_DOUBLE_EQ(viscosityFunction.propertySize, 1);
    ASSERT_DOUBLE_EQ(diffusivityFunction.propertySize, 1);
}

TEST(ConstantTransportTests, ShouldReturnNullFunctionsIfAllValuesZeroForTemperatureFunction) {
    // ARRANGE
    auto constantModel = std::make_shared<ablate::eos::transport::Constant>();

    // ACT
    // ASSERT
    ASSERT_TRUE(constantModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, {}).function == nullptr);
    ASSERT_TRUE(constantModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, {}).function == nullptr);
    ASSERT_TRUE(constantModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, {}).function == nullptr);
}
//
// Created by owen on 5/19/22.
//
#include "eos/radiationProperties/radiationConstant.hpp"
#include "gtest/gtest.h"


TEST(RadiationConstantTests, ShouldRecordConstantValuesForDirectFunction) {
    // ARRANGE
    const PetscReal expectedAbsorptivity = 1;

    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Constant>(expectedAbsorptivity);

    auto absorptivityFunction = constantModel->GetRadiationPropertiesFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, {});

    // ACT
    PetscReal computedAbsorptivity = NAN;

    absorptivityFunction.function(nullptr, &computedAbsorptivity, absorptivityFunction.context.get());

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedAbsorptivity, computedAbsorptivity);
}

TEST(ConstantTransportTests, ShouldReturnNullFunctionsIfAllValuesZeroForDirectFunction) {
    // ARRANGE
    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Constant>();

    // ACT
    // ASSERT
    ASSERT_TRUE(constantModel->GetRadiationPropertiesFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, {}).function == nullptr);
}

TEST(ConstantTransportTests, ShouldRecordConstantValuesForTemperatureFunction) {
    // ARRANGE
    const PetscReal expectedAbsorptivity = 1;

    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Constant>(expectedAbsorptivity);

    auto conductivityFunction = constantModel->GetRadiationPropertiesTemperatureFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, {});

    // ACT
    PetscReal computedAbsorptivity = NAN;

    conductivityFunction.function(nullptr, NAN, &computedAbsorptivity, conductivityFunction.context.get());

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedAbsorptivity, computedAbsorptivity);
}

TEST(ConstantTransportTests, ShouldReturnNullFunctionsIfAllValuesZeroForTemperatureFunction) {
    // ARRANGE
    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Constant>();

    // ACT
    // ASSERT
    ASSERT_TRUE(constantModel->GetRadiationPropertiesTemperatureFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, {}).function == nullptr);
}
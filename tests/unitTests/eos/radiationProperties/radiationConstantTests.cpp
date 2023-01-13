#include "eos/radiationProperties/constant.hpp"
#include "gtest/gtest.h"

TEST(RadiationConstantTests, ShouldRecordConstantValuesForDirectRadiationFunction) {
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

TEST(ConstantTransportTests, ShouldRecordConstantValuesForRadiationTemperatureFunction) {
    // ARRANGE
    const PetscReal expectedAbsorptivity = 1;

    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Constant>(expectedAbsorptivity);

    auto absorptivityFunction = constantModel->GetRadiationPropertiesTemperatureFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, {});

    // ACT
    PetscReal computedAbsorptivity = NAN;

    absorptivityFunction.function(nullptr, NAN, &computedAbsorptivity, absorptivityFunction.context.get());

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedAbsorptivity, computedAbsorptivity);
}
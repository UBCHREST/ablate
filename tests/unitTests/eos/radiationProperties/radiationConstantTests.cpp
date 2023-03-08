#include "eos/mockEOS.hpp"
#include "eos/perfectGas.hpp"
#include "eos/radiationProperties/constant.hpp"
#include "gtest/gtest.h"

TEST(RadiationConstantTests, ShouldRecordConstantValuesForDirectRadiationFunction) {
    // ARRANGE
    const PetscReal expectedAbsorptivity = 1;

    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();  //!< Create a mock eos with parameters to feed to the model.

    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Constant>(eos, expectedAbsorptivity, 1);

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

    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();  //!< Create a mock eos with parameters to feed to the model.

    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Constant>(eos, expectedAbsorptivity, 1);

    auto absorptivityFunction = constantModel->GetRadiationPropertiesTemperatureFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, {});

    // ACT
    PetscReal computedAbsorptivity = NAN;

    absorptivityFunction.function(nullptr, NAN, &computedAbsorptivity, absorptivityFunction.context.get());

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedAbsorptivity, computedAbsorptivity);
}
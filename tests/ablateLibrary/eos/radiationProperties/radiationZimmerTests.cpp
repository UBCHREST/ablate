#include "eos/radiationProperties/radiationZimmer.hpp"
#include "gtest/gtest.h"

struct ZimmerTestParameters {
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal cpIn;

    PetscReal expectedConductivity;
    PetscReal expectedViscosity;
    PetscReal expectedDiffusivity;
};

class ZimmerTestFixture : public ::testing::TestWithParam<ZimmerTestParameters> {};

TEST(ZimmerTestFixture, ShouldProduceExpectedValuesForField) {
    // ARRANGE
    const PetscReal expectedAbsorptivity = 1;                                                  //!< The absorptivity that we expect from the set field of components
    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Zimmer>(nullptr);  //!< An instantiation of the Zimmer model (with options set to nullptr)
    auto absorptivityFunction = constantModel->GetRadiationPropertiesFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, {});

    /** This section should set the fields with a certain distribution of material such that the absorptivity of that field produces a specific result */

    // ACT
    PetscReal computedAbsorptivity = NAN;                                                               //!< Declaration of the computed absorptivity
    absorptivityFunction.function(nullptr, &computedAbsorptivity, absorptivityFunction.context.get());  //!< Getting the absorptivity from the density, temperature, and mass fraction fields

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedAbsorptivity, computedAbsorptivity);  //!< Comparing the values between the expected and the computed absorptivity
}

TEST(ZimmerTestFixture, ShouldProduceOtherExpectedValuesForField) {
    // ARRANGE
    const PetscReal expectedAbsorptivity = 10;                                                 //!< The absorptivity that we expect from the set field of components
    auto constantModel = std::make_shared<ablate::eos::radiationProperties::Zimmer>(nullptr);  //!< An instantiation of the Zimmer model (with options set to nullptr)
    auto absorptivityFunction = constantModel->GetRadiationPropertiesTemperatureFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, {});

    /** This section should set the fields with a certain distribution of material such that the absorptivity of that field produces a specific result */

    // ACT
    PetscReal computedAbsorptivity = NAN;                                                                    //!< Declaration of the computed absorptivity
    absorptivityFunction.function(nullptr, NAN, &computedAbsorptivity, absorptivityFunction.context.get());  //!< Getting the absorptivity from the density, temperature, and mass fraction fields

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedAbsorptivity, computedAbsorptivity);  //!< Comparing the values between the expected and the computed absorptivity
}

INSTANTIATE_TEST_SUITE_P(RadationZimmerTests, ZimmerTestFixture,
                         testing::Values((ZimmerTestParameters){
                             .temperatureIn = 300.0, .densityIn = 1.1, .cpIn = 1001.1, .expectedConductivity = 0.02615186, .expectedViscosity = 1.8469051E-5, .expectedDiffusivity = 2.374829E-5}));

#include <eos/mockEOS.hpp>
#include "eos/radiationProperties/sootSpectrumAbsorption.hpp"
#include "gtest/gtest.h"

struct SootSpectrumTestParameters {
    std::vector<ablate::domain::Field> fields;
    std::vector<PetscReal> conservedValues;
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal expectedAbsorptivity;
};

class SootSpectrumTestFixture : public ::testing::TestWithParam<SootSpectrumTestParameters> {};

TEST_P(SootSpectrumTestFixture, ShouldProduceExpectedValuesForField) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();  //!< Create a mock eos with parameters to feed to the Zimmer model.
    /** Input values for the mock eos to carry into the Zimer model. This will require values for each of the fields. */
    EXPECT_CALL(*eos, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(
            ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = SootSpectrumTestFixture::GetParam().temperatureIn; })));
    EXPECT_CALL(*eos, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::Density, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(
            [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = SootSpectrumTestFixture::GetParam().densityIn; })));

    //! Later tests can be added to deal with the even distribution. For now this will not be used.
    std::vector<double> wavelengths = {650.E-9, 532.E-9, 470.E-9};
    auto sootModel = std::make_shared<ablate::eos::radiationProperties::SootSpectrumAbsorption>(eos, 0, 0, 0, wavelengths);  //!< An instantiation of the Zimmer model (with options set to nullptr)
    auto absorptivityFunction = sootModel->GetRadiationPropertiesFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, SootSpectrumTestFixture::GetParam().fields);

    /** This section should set the fields with a certain distribution of material such that the absorptivity of that field produces a specific result */

    // ACT
    PetscReal computedAbsorptivity = NAN;  //!< Declaration of the computed absorptivity
    absorptivityFunction.function(SootSpectrumTestFixture::GetParam().conservedValues.data(),
                                  &computedAbsorptivity,
                                  absorptivityFunction.context.get());  //!< Getting the absorptivity from the density, temperature, and mass fraction fields

    // ASSERT
    ASSERT_NEAR(SootSpectrumTestFixture::GetParam().expectedAbsorptivity, computedAbsorptivity, 1E-5);  //!< Comparing the values between the expected and the computed absorptivity
}

INSTANTIATE_TEST_SUITE_P(RadationSootTests, SootSpectrumTestFixture,
                         testing::Values(/** A test with all valid species for the Soot model */
                                         (SootSpectrumTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                                                                 ablate::domain::Field{.name = "densityYi", .numberComponents = 1, .components = {"C_solid"}, .offset = 5}},
                                                                      .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.0025},  //!< The Density Yi values live here
                                                                      .temperatureIn = 300.0,
                                                                      .densityIn = 0.01,  //!< The density is read through the equation of state above, not here
                                                                      .expectedAbsorptivity = 8.1987145931002505},
                                         /** A test with three valid species for the Soot model. */
                                         (SootSpectrumTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                                                                 ablate::domain::Field{.name = "densityYi", .numberComponents = 1, .components = {"C_solid"}, .offset = 5}},
                                                                      .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.0025},  //!< The Density Yi values live here
                                                                      .temperatureIn = 800.0,
                                                                      .densityIn = 1.1,
                                                                      .expectedAbsorptivity = 8.1987145931002505},
                                         /** A test with one valid species for the Soot model. */
                                         (SootSpectrumTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 1, .offset = 0},
                                                                                 ablate::domain::Field{.name = "densityYi", .numberComponents = 4, .components = {"C_solid"}, .offset = 5}},
                                                                      .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.05},  //!< The Density Yi values live here
                                                                      .temperatureIn = 1200.0,
                                                                      .densityIn = 1.1,
                                                                      .expectedAbsorptivity = 163.97429186200497}));
#include <eos/mockEOS.hpp>
#include "eos/radiationProperties/zimmer.hpp"
#include "gtest/gtest.h"

struct ZimmerTestParameters {
    std::vector<ablate::domain::Field> fields;
    std::vector<PetscReal> conservedValues;
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal expectedAbsorptivity;
};

class ZimmerTestFixture : public ::testing::TestWithParam<ZimmerTestParameters> {};

TEST_P(ZimmerTestFixture, ShouldProduceExpectedValuesForField) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();  //!< Create a mock eos with parameters to feed to the Zimmer model.
    /** Input values for the mock eos to carry into the Zimer model. This will require values for each of the fields. */
    EXPECT_CALL(*eos, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(
            ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = ZimmerTestFixture::GetParam().temperatureIn; })));
    EXPECT_CALL(*eos, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::Density, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(
            [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = ZimmerTestFixture::GetParam().densityIn; })));

    auto zimmerModel = std::make_shared<ablate::eos::radiationProperties::Zimmer>(eos);  //!< An instantiation of the Zimmer model (with options set to nullptr)
    auto absorptivityFunction = zimmerModel->GetRadiationPropertiesFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, ZimmerTestFixture::GetParam().fields);

    /** This section should set the fields with a certain distribution of material such that the absorptivity of that field produces a specific result */

    // ACT
    PetscReal computedAbsorptivity = NAN;  //!< Declaration of the computed absorptivity
    absorptivityFunction.function(ZimmerTestFixture::GetParam().conservedValues.data(),
                                  &computedAbsorptivity,
                                  absorptivityFunction.context.get());  //!< Getting the absorptivity from the density, temperature, and mass fraction fields

    // ASSERT
    ASSERT_NEAR(ZimmerTestFixture::GetParam().expectedAbsorptivity, computedAbsorptivity, 1E-5);  //!< Comparing the values between the expected and the computed absorptivity
}

INSTANTIATE_TEST_SUITE_P(RadationZimmerTests, ZimmerTestFixture,
                         testing::Values(/** A test with all valid species for the Zimmer model */
                                         (ZimmerTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                                                           ablate::domain::Field{.name = "densityYi", .numberComponents = 4, .components = {"H2O", "co2", "CH4", "co"}, .offset = 5}},
                                                                .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.0025, 0.0025, 0.0025, 0.0025},  //!< The Density Yi values live here
                                                                .temperatureIn = 300.0,
                                                                .densityIn = 0.01,  //!< The density is read through the equation of state above, not here
                                                                .expectedAbsorptivity = 0.268109},
                                         /** A test with three valid species for the Zimmer model. */
                                         (ZimmerTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                                                           ablate::domain::Field{.name = "densityYi", .numberComponents = 4, .components = {"h2o", "CO2", "ch4"}, .offset = 5}},
                                                                .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.0025, 0.0025, 0.0025},  //!< The Density Yi values live here
                                                                .temperatureIn = 300.0,
                                                                .densityIn = 1.1,
                                                                .expectedAbsorptivity = 0.266579961079},
                                         /** A test with one valid species for the Zimmer model. */
                                         (ZimmerTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                                                           ablate::domain::Field{.name = "densityYi", .numberComponents = 4, .components = {"h2o"}, .offset = 5}},
                                                                .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.0025},  //!< The Density Yi values live here
                                                                .temperatureIn = 300.0,
                                                                .densityIn = 1.1,
                                                                .expectedAbsorptivity = 0.1230770},
                                         /** A test with all valid species producing different results*/
                                         (ZimmerTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                                                           ablate::domain::Field{.name = "densityYi", .numberComponents = 4, .components = {"H2O", "co2", "ch4", "co"}, .offset = 5}},
                                                                .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.0045, 0.0065, 0.0085, 0.0025},  //!< The Density Yi values live here
                                                                .temperatureIn = 300.0,
                                                                .densityIn = 0.01,
                                                                .expectedAbsorptivity = 0.6132735699},
                                         /** A test with all valid species producing other different results*/
                                         (ZimmerTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                                                           ablate::domain::Field{.name = "densityYi", .numberComponents = 4, .components = {"h2o", "co2", "ch4", "CO"}, .offset = 5}},
                                                                .conservedValues = {0.5, NAN, NAN, NAN, NAN, 0.0015, 0.0005, 0.0035, 0.0045},  //!< The Density Yi values live here
                                                                .temperatureIn = 1200.0,
                                                                .densityIn = 1.1,
                                                                .expectedAbsorptivity = 0.2480760899},
                                         (ZimmerTestParameters){.fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                                                           ablate::domain::Field{.name = "densityYi", .numberComponents = 4, .components = {"H2O", "co2", "ch4", "co"}, .offset = 5}},
                                                                .conservedValues = {0.00, NAN, NAN, NAN, NAN, 0.0, 0.0, 0.0, 0.0},  //!< The Density Yi values live here
                                                                .temperatureIn = 300.0,
                                                                .densityIn = 0.00,
                                                                .expectedAbsorptivity = 0.0}));

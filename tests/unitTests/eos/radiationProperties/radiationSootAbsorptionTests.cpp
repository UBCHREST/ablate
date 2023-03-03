#include <eos/mockEOS.hpp>
#include "domain/mockField.hpp"
#include "eos/radiationProperties/sootMeanAbsorption.hpp"
#include "eos/tChemSoot.hpp"
#include "gtest/gtest.h"

struct SootTestParameters {
    std::vector<ablate::domain::Field> fields;
    std::vector<PetscReal> conservedValues;
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal expectedAbsorptivity;
};

class SootTestFixture : public ::testing::TestWithParam<SootTestParameters> {};

TEST_P(SootTestFixture, ShouldProduceExpectedValuesForField) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();  //!< Create a mock eos with parameters to feed to the Zimmer model.
    /** Input values for the mock eos to carry into the Zimer model. This will require values for each of the fields. */
    EXPECT_CALL(*eos, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(
            ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = SootTestFixture::GetParam().temperatureIn; })));
    EXPECT_CALL(*eos, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::Density, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(
            [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = SootTestFixture::GetParam().densityIn; })));

    auto sootModel = std::make_shared<ablate::eos::radiationProperties::SootMeanAbsorption>(eos);  //!< An instantiation of the Zimmer model (with options set to nullptr)
    auto absorptivityFunction = sootModel->GetAbsorptionPropertiesFunction(ablate::eos::radiationProperties::RadiationProperty::Absorptivity, SootTestFixture::GetParam().fields);

    /** This section should set the fields with a certain distribution of material such that the absorptivity of that field produces a specific result */

    // ACT
    PetscReal computedAbsorptivity = NAN;  //!< Declaration of the computed absorptivity
    absorptivityFunction.function(SootTestFixture::GetParam().conservedValues.data(),
                                  &computedAbsorptivity,
                                  absorptivityFunction.context.get());  //!< Getting the absorptivity from the density, temperature, and mass fraction fields

    // ASSERT
    ASSERT_NEAR(SootTestFixture::GetParam().expectedAbsorptivity, computedAbsorptivity, 1E-5);  //!< Comparing the values between the expected and the computed absorptivity
}

INSTANTIATE_TEST_SUITE_P(RadationSootTests, SootTestFixture,
                         testing::Values(/** A test with all valid species for the Soot model */
                                         (SootTestParameters){.fields = {ablateTesting::domain::MockField::Create("euler", 5, 0),
                                                                         ablateTesting::domain::MockField::Create("densityYi", {ablate::eos::TChemSoot::CSolidName}, 5)},
                                                              .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.0025},  //!< The Density Yi values live here
                                                              .temperatureIn = 300.0,
                                                              .densityIn = 0.01,  //!< The density is read through the equation of state above, not here
                                                              .expectedAbsorptivity = 0.67868822124163297},
                                         /** A test with three valid species for the Soot model. */
                                         (SootTestParameters){.fields = {ablateTesting::domain::MockField::Create("euler", 5, 0),
                                                                         ablateTesting::domain::MockField::Create("densityYi", {ablate::eos::TChemSoot::CSolidName}, 5)},
                                                              .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.0025},  //!< The Density Yi values live here
                                                              .temperatureIn = 800.0,
                                                              .densityIn = 1.1,
                                                              .expectedAbsorptivity = 1.8098352566443547},
                                         /** A test with one valid species for the Soot model. */
                                         (SootTestParameters){.fields = {ablateTesting::domain::MockField::Create("euler", 1, 0),
                                                                         ablateTesting::domain::MockField::Create("densityYi", {ablate::eos::TChemSoot::CSolidName, "", "", ""}, 5)},
                                                              .conservedValues = {0.01, NAN, NAN, NAN, NAN, 0.05},  //!< The Density Yi values live here
                                                              .temperatureIn = 1200.0,
                                                              .densityIn = 1.1,
                                                              .expectedAbsorptivity = 54.295057699330641}));
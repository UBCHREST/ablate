#include "PetscTestFixture.hpp"
#include "eos/perfectGas.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS create and view tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestCreateAndViewParameters {
    std::map<std::string, std::string> options;
    std::vector<std::string> species = {};
    std::string expectedView;
};

class PerfectGasTestCreateAndViewFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestCreateAndViewParameters> {};

TEST_P(PerfectGasTestCreateAndViewFixture, ShouldCreateAndView) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters, GetParam().species);

    std::stringstream outputStream;

    // act
    outputStream << *eos;

    // assert the output is as expected
    auto outputString = outputStream.str();
    ASSERT_EQ(outputString, GetParam().expectedView);
}

INSTANTIATE_TEST_SUITE_P(PerfectGasEOSTests, PerfectGasTestCreateAndViewFixture,
                         testing::Values((EOSTestCreateAndViewParameters){.options = {}, .expectedView = "EOS: perfectGas\n\tgamma: 1.4\n\tRgas: 287\n"},
                                         (EOSTestCreateAndViewParameters){.options = {{"gamma", "3.2"}, {"Rgas", "100.2"}}, .expectedView = "EOS: perfectGas\n\tgamma: 3.2\n\tRgas: 100.2\n"},
                                         (EOSTestCreateAndViewParameters){.options = {{"gamma", "3.2"}, {"Rgas", "100.2"}},
                                                                          .species = {"O2", "N2"},
                                                                          .expectedView = "EOS: perfectGas\n\tgamma: 3.2\n\tRgas: 100.2\n\tspecies: O2, N2\n"}),
                         [](const testing::TestParamInfo<EOSTestCreateAndViewParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS Thermodynamic property tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct PGTestParameters {
    std::map<std::string, std::string> options;
    std::vector<std::string> species = {};
    ablate::eos::ThermodynamicProperty thermodynamicProperty;
    std::vector<ablate::domain::Field> fields;
    std::vector<PetscReal> conservedValues;
    std::optional<PetscReal> expectedTemperature;
    std::vector<PetscReal> expectedValue;
};

class PGThermodynamicPropertyTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<PGTestParameters> {};

TEST_P(PGThermodynamicPropertyTestFixture, ShouldComputeProperty) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters, GetParam().species);

    // get the test params
    const auto& params = GetParam();

    // act/assert check for compute without temperature
    auto thermodynamicFunction = eos->GetThermodynamicFunction(params.thermodynamicProperty, params.fields);
    std::vector<PetscReal> computedProperty(params.expectedValue.size(), NAN);
    PetscErrorCode ierr = thermodynamicFunction.function(params.conservedValues.data(), computedProperty.data(), thermodynamicFunction.context.get());
    ASSERT_EQ(ierr, 0);
    for (std::size_t c = 0; c < params.expectedValue.size(); c++) {
        ASSERT_NEAR(computedProperty[c], params.expectedValue[c], 1E-6) << "for direct function ";
    }
    // act/assert check for compute when temperature is known
    auto temperatureFunction = eos->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, params.fields);
    PetscReal computedTemperature;
    ierr = temperatureFunction.function(params.conservedValues.data(), &computedTemperature, temperatureFunction.context.get());
    ASSERT_EQ(ierr, 0);

    if (params.expectedTemperature) {
        ASSERT_NEAR(computedTemperature, params.expectedTemperature.value(), 1E-6) << "for computed temperature ";
    }

    auto thermodynamicTemperatureFunction = eos->GetThermodynamicTemperatureFunction(params.thermodynamicProperty, params.fields);
    computedProperty = std::vector<PetscReal>(params.expectedValue.size(), NAN);
    ierr = thermodynamicTemperatureFunction.function(params.conservedValues.data(), computedTemperature, computedProperty.data(), thermodynamicTemperatureFunction.context.get());

    ASSERT_EQ(ierr, 0);
    for (std::size_t c = 0; c < params.expectedValue.size(); c++) {
        ASSERT_NEAR(computedProperty[c], params.expectedValue[c], 1E-6) << " for temperature function ";
    }
}

INSTANTIATE_TEST_SUITE_P(PerfectGasEOSTests, PGThermodynamicPropertyTestFixture,
                         testing::Values((PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Pressure,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {1.2, 1.2 * 1E5, 1.2 * 10, 1.2 * -20, 1.2 * 30},
                                                            .expectedValue = {47664}},
                                         (PGTestParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Pressure,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 0.9, 0.9 * 1.56E5, 0.0},
                                                            .expectedValue = {140400}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {1.2, 1.2 * 1E5, 1.2 * 10, 1.2 * -20, 1.2 * 30},
                                                            .expectedValue = {99300}},
                                         (PGTestParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 0.9, 0.9 * 1.56E5, 0.0},
                                                            .expectedValue = {1.56E+05}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeedOfSound,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {1.2, 1.2 * 1E5, 1.2 * 10, 1.2 * -20, 1.2 * 30},
                                                            .expectedValue = {235.8134856}},
                                         (PGTestParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeedOfSound,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 0.9, 0.9 * 1.56E5, 0.0},
                                                            .expectedValue = {558.5696018}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Temperature,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {1.2, 1.2 * 1.50E5, 1.2 * 10, 1.2 * -20, 1.2 * 30},
                                                            .expectedValue = {208.0836237}},
                                         (PGTestParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Temperature,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 0.9, 0.9 * 1.56E5, 0.0},
                                                            .expectedValue = {39000}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {1.2, 1.2 * 1.50E5, 1.2 * 10, 1.2 * -20, 1.2 * 30},
                                                            .expectedValue = {1004.5}},
                                         (PGTestParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 0.9, 0.9 * 1.56E5, 0.0},
                                                            .expectedValue = {8.0}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {1.2, 1.2 * 1.50E5, 1.2 * 10, 1.2 * -20, 1.2 * 30},
                                                            .expectedValue = {1004.5 / 1.4}},
                                         (PGTestParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 0.9, 0.9 * 1.56E5, 0.0},
                                                            .expectedValue = {8.0 / 2.0}},
                                         (PGTestParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {.9, .9 * 1.567E+05, .9 * 10, .9 * -20, .9 * 30},
                                                            .expectedTemperature = 39000,
                                                            .expectedValue = {1.56E5}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {1.1, 1.1 * 2.51825E+05, 1.1 * 10, 1.1 * -20, 1.1 * 30},
                                                            .expectedTemperature = 350.0,
                                                            .expectedValue = {251125.00}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {20.1, 20.1 * 2.51825E+05, 20.1 * 10, 20.1 * -20, 20.1 * 30},
                                                            .expectedTemperature = 350.0,
                                                            .expectedValue = {251125.00}},

                                         (PGTestParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SensibleEnthalpy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {.9, .9 * 1.567E+05, .9 * 10, .9 * -20, .9 * 30},
                                                            .expectedTemperature = 39000,
                                                            .expectedValue = {312000}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SensibleEnthalpy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {1.1, 1.1 * 2.51825E+05, 1.1 * 10, 1.1 * -20, 1.1 * 30},
                                                            .expectedTemperature = 350.0,
                                                            .expectedValue = {351575.00}},
                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SensibleEnthalpy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {20.1, 20.1 * 2.51825E+05, 20.1 * 10, 20.1 * -20, 20.1 * 30},
                                                            .expectedTemperature = 350.0,
                                                            .expectedValue = {351575.00}},

                                         (PGTestParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                            .species = std::vector<std::string>{"O2", "CH4", "N2"},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {20.1, 20.1 * 2.51825E+05, 20.1 * 10, 20.1 * -20, 20.1 * 30},
                                                            .expectedValue = std::vector<PetscReal>{0.0, 0.0, 0.0}}),

                         [](const testing::TestParamInfo<PGTestParameters>& info) { return std::to_string(info.index) + "_" + std::string(to_string(info.param.thermodynamicProperty)); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS get species tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(PerfectGasEOSTests, PerfectGasShouldReportNoSpeciesByDefault) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters);

    // act
    auto species = eos->GetSpecies();

    // assert
    ASSERT_EQ(0, species.size());
}

TEST(PerfectGasEOSTests, PerfectGasShouldReportSpeciesWhenProvided) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters, std::vector<std::string>{"N2", "H2"});

    // act
    auto species = eos->GetSpecies();

    // assert
    ASSERT_EQ(2, species.size());
    ASSERT_EQ("N2", species[0]);
    ASSERT_EQ("H2", species[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perfect Gas DensityFunctionFromTemperaturePressure
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct PerfectGasTestComputeDensityParameters {
    std::map<std::string, std::string> options;
    PetscReal temperatureIn;
    PetscReal pressureIn;
    PetscReal expectedDensity;
};

class PerfectGasTestComputeDensityTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<PerfectGasTestComputeDensityParameters> {};

TEST_P(PerfectGasTestComputeDensityTestFixture, ShouldComputeCorrectTemperature) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters);

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal density;

    // act
    PetscErrorCode ierr =
        eos->GetComputeDensityFunctionFromTemperaturePressureFunction()(params.temperatureIn, params.pressureIn, nullptr, &density, eos->GetComputeDensityFunctionFromTemperaturePressureContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(density, params.expectedDensity, 1E-3);
}

INSTANTIATE_TEST_SUITE_P(
    PerfectGasEOSTests, PerfectGasTestComputeDensityTestFixture,
    testing::Values((PerfectGasTestComputeDensityParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}}, .temperatureIn = 300.0, .pressureIn = 101325.0, .expectedDensity = 1.17682},
                    (PerfectGasTestComputeDensityParameters){.options = {{"gamma", "1.4"}, {"Rgas", "487.0"}}, .temperatureIn = 1000.0, .pressureIn = 1013250.0, .expectedDensity = 2.08059}),
    [](const testing::TestParamInfo<PerfectGasTestComputeDensityParameters>& info) { return std::to_string(info.index); });

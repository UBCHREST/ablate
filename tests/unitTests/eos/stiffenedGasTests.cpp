#include "PetscTestFixture.hpp"
#include "eos/stiffenedGas.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS create and view tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct StiffenedGasEOSTestCreateAndViewParameters {
    std::map<std::string, std::string> options;
    std::vector<std::string> species = {};
    std::string expectedView;
};

class StiffenedGasTestCreateAndViewFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<StiffenedGasEOSTestCreateAndViewParameters> {};

TEST_P(StiffenedGasTestCreateAndViewFixture, ShouldCreateAndView) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters, GetParam().species);

    std::stringstream outputStream;

    // act
    outputStream << *eos;

    // assert the output is as expected
    auto outputString = outputStream.str();
    ASSERT_EQ(outputString, GetParam().expectedView);
}

INSTANTIATE_TEST_SUITE_P(StiffenedGasEOSTests, StiffenedGasTestCreateAndViewFixture,
                         testing::Values((StiffenedGasEOSTestCreateAndViewParameters){.options = {}, .expectedView = "EOS: stiffenedGas\n\tgamma: 1.932\n\tCp: 8095.08\n\tp0: 1.1645e+09\n"},
                                         (StiffenedGasEOSTestCreateAndViewParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                                                      .expectedView = "EOS: stiffenedGas\n\tgamma: 3.2\n\tCp: 100.2\n\tp0: 3.5e+06\n"},
                                         (StiffenedGasEOSTestCreateAndViewParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},  // need to replace with real state for expected
                                                                                                                                                        // internal energy, speed of sound, pressure
                                                                                      .species = {"O2", "N2"},
                                                                                      .expectedView = "EOS: stiffenedGas\n\tgamma: 3.2\n\tCp: 100.2\n\tp0: 3.5e+06\n\tspecies: O2, N2\n"}),
                         [](const testing::TestParamInfo<StiffenedGasEOSTestCreateAndViewParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS Thermodynamic property tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct SGTestParameters {
    std::map<std::string, std::string> options;
    std::vector<std::string> species = {};
    ablate::eos::ThermodynamicProperty thermodynamicProperty;
    std::vector<ablate::domain::Field> fields;
    std::vector<PetscReal> conservedValues;
    std::optional<PetscReal> expectedTemperature;
    std::vector<PetscReal> expectedValue;
};

class SGThermodynamicPropertyTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<SGTestParameters> {};

TEST_P(SGThermodynamicPropertyTestFixture, ShouldComputeProperty) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters, GetParam().species);

    // get the test params
    const auto& params = GetParam();

    // act/assert check for compute without temperature
    auto thermodynamicFunction = eos->GetThermodynamicFunction(params.thermodynamicProperty, params.fields);
    std::vector<PetscReal> computedProperty(params.expectedValue.size(), NAN);
    ASSERT_EQ(0, thermodynamicFunction.function(params.conservedValues.data(), computedProperty.data(), thermodynamicFunction.context.get()));
    for (std::size_t c = 0; c < params.expectedValue.size(); c++) {
        ASSERT_NEAR(computedProperty[c], params.expectedValue[c], 1E-6) << "for direct function ";
    }
    // act/assert check for compute when temperature is known
    auto temperatureFunction = eos->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, params.fields);
    PetscReal computedTemperature;
    ASSERT_EQ(0, temperatureFunction.function(params.conservedValues.data(), &computedTemperature, temperatureFunction.context.get()));
    ASSERT_EQ(1, temperatureFunction.propertySize) << "The temperature property size should be 1";

    if (params.expectedTemperature) {
        ASSERT_NEAR(computedTemperature, params.expectedTemperature.value(), 1E-6) << "for computed temperature ";
    }

    auto thermodynamicTemperatureFunction = eos->GetThermodynamicTemperatureFunction(params.thermodynamicProperty, params.fields);
    computedProperty = std::vector<PetscReal>(params.expectedValue.size(), NAN);
    ASSERT_EQ(params.expectedValue.size(), thermodynamicFunction.propertySize) << "The " << params.thermodynamicProperty << " property size should be " << params.expectedValue.size();
    ASSERT_EQ(0, thermodynamicTemperatureFunction.function(params.conservedValues.data(), computedTemperature, computedProperty.data(), thermodynamicTemperatureFunction.context.get()));
    for (std::size_t c = 0; c < params.expectedValue.size(); c++) {
        ASSERT_NEAR(computedProperty[c], params.expectedValue[c], 1E-6) << " for temperature function ";
    }
}

INSTANTIATE_TEST_SUITE_P(StiffenedGasEOSTests, SGThermodynamicPropertyTestFixture,
                         testing::Values((SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {998.7, 998.7 * 2.5E6, 998.7 * 10, 998.7 * -20, 998.7 * 30},
                                                            .expectedValue = {2499300.0}},
                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeedOfSound,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {998.7, 998.7 * 2.5E6, 998.7 * 10, 998.7 * -20, 998.7 * 30},
                                                            .expectedValue = {1549.4332810120738}},
                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Pressure,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {998.7, 998.7 * 2.5E6, 998.7 * 10, 998.7 * -20, 998.7 * 30},
                                                            .expectedValue = {76505448.11999989}},
                                         (SGTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 800, 800 * 1.2E5, 0.0},
                                                            .expectedValue = {1.2E+05}},
                                         (SGTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeedOfSound,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 800, 800 * 1.2E5, 0.0},
                                                            .expectedValue = {902.2194855}},
                                         (SGTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Density,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 800, 800 * 1.2E5, 0.0},
                                                            .expectedValue = {800}},
                                         (SGTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Pressure,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 800, 800 * 1.2E5, 0.0},
                                                            .expectedValue = {2e8}},

                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Temperature,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {998.7, 998.7 * 2.5E+06, 998.7 * 10, 998.7 * 20, 998.7 * 30},
                                                            .expectedValue = {318.2062480747645}},
                                         (SGTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Temperature,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 800, 800 * 1.2E5, 0.0},
                                                            .expectedValue = {3692.6147704590817}},

                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {998.7, 998.7 * 2.63321582E+06, 998.7 * 10, 998.7 * 20, 998.7 * 30},
                                                            .expectedTemperature = 350.0,
                                                            .expectedValue = {2632515.82}},
                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 20.1, 20.1 * 5.9401823383084600E+07, 0.0},
                                                            .expectedTemperature = 350.0,
                                                            .expectedValue = {59401823.38308461}},
                                         (SGTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SensibleEnthalpy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {800, 800 * 1.2262625000000000E+06, 800 * 10, 800 * 20, 800 * 30},
                                                            .expectedTemperature = 39000.0,
                                                            .expectedValue = {3907800.0}},
                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SensibleEnthalpy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 2}},
                                                            .conservedValues = {NAN, NAN, 998.7, 998.7 * 2.6332158205667400E+06, 998.7 * 10, 998.7 * 20, 998.7 * 30},
                                                            .expectedTemperature = 350.0,
                                                            .expectedValue = {2833278.0}},

                                         (SGTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {800, 800 * 1.2262625000000000E+06, 800 * 10, 800 * 20, 800 * 30},
                                                            .expectedValue = {100.2}},
                                         (SGTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {800, 800 * 1.2262625000000000E+06, 800 * 10, 800 * 20, 800 * 30},
                                                            .expectedValue = {31.3125}},
                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {800, 800 * 1.2262625000000000E+06, 800 * 10, 800 * 20, 800 * 30},
                                                            .expectedValue = {8095.08}},
                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {800, 800 * 1.2262625000000000E+06, 800 * 10, 800 * 20, 800 * 30},
                                                            .expectedValue = {4190.0}},

                                         (SGTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                                            .species = std::vector<std::string>{"O2", "CH4", "N2"},
                                                            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
                                                            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}},
                                                            .conservedValues = {20.1, 20.1 * 2.51825E+05, 20.1 * 10, 20.1 * -20, 20.1 * 30},
                                                            .expectedValue = std::vector<PetscReal>{0.0, 0.0, 0.0}}

                                         ),

                         [](const testing::TestParamInfo<SGTestParameters>& info) { return std::to_string(info.index) + "_" + std::string(to_string(info.param.thermodynamicProperty)); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS get species tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(StiffenedGasEOSTests, StiffenedGasShouldReportNoSpeciesEctByDefault) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters);

    // act
    ASSERT_EQ(0, eos->GetSpeciesVariables().size());
    ASSERT_EQ(0, eos->GetFieldFunctionProperties().size());
    ASSERT_EQ(0, eos->GetProgressVariables().size());
}

TEST(StiffenedGasEOSTests, StiffenedGasShouldReportSpeciesWhenProvided) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters, std::vector<std::string>{"N2", "H2"});

    // act
    auto species = eos->GetSpeciesVariables();

    // assert
    ASSERT_EQ(2, species.size());
    ASSERT_EQ("N2", species[0]);
    ASSERT_EQ("H2", species[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Stiffened Gas FieldFunctionTests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct SGFieldFunctionTestParameters {
    // eos init
    std::map<std::string, std::string> options;
    std::vector<std::string> species = {};

    // field function init
    std::string field;
    ablate::eos::ThermodynamicProperty property1;
    ablate::eos::ThermodynamicProperty property2;

    // inputs
    PetscReal property1Value;
    PetscReal property2Value;
    std::vector<PetscReal> velocity;
    std::vector<PetscReal> yi;
    std::vector<PetscReal> expectedValue;
};

class SGFieldFunctionTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<SGFieldFunctionTestParameters> {};

TEST_P(SGFieldFunctionTestFixture, ShouldComputeField) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters, GetParam().species);

    // get the test params
    const auto& params = GetParam();
    std::vector<PetscReal> actualValue(params.expectedValue.size(), NAN);

    // act
    auto stateFunction = eos->GetFieldFunctionFunction(params.field, params.property1, params.property2, {ablate::eos::EOS::YI});
    stateFunction(params.property1Value, params.property2Value, params.velocity.size(), params.velocity.data(), params.yi.data(), actualValue.data());

    // assert
    for (std::size_t c = 0; c < params.expectedValue.size(); c++) {
        ASSERT_NEAR(actualValue[c], params.expectedValue[c], 1E-3) << "for component[" << c << "] ";
    }
}

INSTANTIATE_TEST_SUITE_P(
    StiffenedGasEOSTests, SGFieldFunctionTestFixture,
    testing::Values(
        (SGFieldFunctionTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                        .field = "euler",
                                        .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property1Value = 300.0,
                                        .property2Value = 101325.0,
                                        .velocity = {10.0, 20, 30},
                                        .expectedValue = {994.090880767274, 994.090880767274 * 2.4291220726994300E+06, 994.090880767274 * 10, 994.090880767274 * 20, 994.090880767274 * 30}},
        (SGFieldFunctionTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                        .field = "euler",
                                        .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property1Value = 1000.0,
                                        .property2Value = 1013250.0,
                                        .velocity = {0.0},
                                        .expectedValue = {65.51624025, 65.51624025 * 8.4734368941450200E+04, 0.0}},

        (SGFieldFunctionTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                        .field = "euler",
                                        .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property1Value = 101325.0,
                                        .property2Value = 300.0,
                                        .velocity = {10.0, 20, 30},
                                        .expectedValue = {994.090880767274, 994.090880767274 * 2.4291220726994300E+06, 994.090880767274 * 10, 994.090880767274 * 20, 994.090880767274 * 30}},
        (SGFieldFunctionTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                        .field = "euler",
                                        .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property1Value = 1013250.0,
                                        .property2Value = 1000.0,
                                        .velocity = {0.0},
                                        .expectedValue = {65.51624025, 65.51624025 * 8.4734368941450200E+04, 0.0}},

        (SGFieldFunctionTestParameters){.options = {{"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}},
                                        .species = {"H2", "O2", "N2"},
                                        .field = "densityYi",
                                        .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property1Value = 300.0,
                                        .property2Value = 101325.0,
                                        .velocity = {10.0, 20, 30},
                                        .yi = {.1, .3, .6},
                                        .expectedValue = {994.090880767274 * .1, 994.090880767274 * .3, 994.090880767274 * 0.6}},
        (SGFieldFunctionTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                        .species = {"H2", "O2", "N2"},
                                        .field = "densityYi",
                                        .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property1Value = 1013250.0,
                                        .property2Value = 1000.0,
                                        .velocity = {0.0},
                                        .yi = {.1, .3, .6},
                                        .expectedValue = {65.51624025 * .1, 65.51624025 * .3, 65.51624025 * 0.6}},

        (SGFieldFunctionTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                        .field = "euler",
                                        .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                        .property1Value = 1013250.0,
                                        .property2Value = 84734.368941450171,
                                        .velocity = {1000.0},
                                        .expectedValue = {65.51624025, 38309597.396116853, 65516.240246779162}},
        (SGFieldFunctionTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                        .field = "euler",
                                        .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                        .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property1Value = 84734.368941450171,
                                        .property2Value = 1013250.0,
                                        .velocity = {0.0, 1000.0},
                                        .expectedValue = {65.51624025, 38309597.396116853, 0.0, 65516.240246779162}},

        (SGFieldFunctionTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                        .species = {"H2", "O2", "N2"},
                                        .field = "densityYi",
                                        .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                        .property1Value = 1013250.0,
                                        .property2Value = 84734.368941450171,
                                        .velocity = {1000.0},
                                        .yi = {.1, .3, .6},
                                        .expectedValue = {65.51624025 * .1, 65.51624025 * .3, 65.51624025 * .6}},
        (SGFieldFunctionTestParameters){.options = {{"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "3.5e6"}},
                                        .species = {"H2", "O2", "N2"},
                                        .field = "densityYi",
                                        .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                        .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property1Value = 84734.368941450171,
                                        .property2Value = 1013250.0,
                                        .velocity = {0.0, 1000.0},
                                        .yi = {.1, .3, .6},
                                        .expectedValue = {65.51624025 * .1, 65.51624025 * .3, 65.51624025 * .6}}),

    [](const testing::TestParamInfo<SGFieldFunctionTestParameters>& info) {
        return std::to_string(info.index) + "_" + info.param.field + "_from_" + std::string(to_string(info.param.property1)) + "_" + std::string(to_string(info.param.property2));
    });

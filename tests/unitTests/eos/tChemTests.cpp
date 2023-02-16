#include <numeric>
#include "PetscTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "eos/tChem.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "solver/dynamicRange.hpp"

/*
 * Helper function to fill mass fraction
 */
static std::vector<PetscReal> GetMassFraction(const std::vector<std::string>& species, const std::map<std::string, PetscReal>& yiIn) {
    std::vector<PetscReal> yi(species.size(), 0.0);

    for (const auto& value : yiIn) {
        // Get the index
        auto it = std::find(species.begin(), species.end(), value.first);
        if (it != species.end()) {
            auto index = std::distance(species.begin(), it);

            yi[index] = value.second;
        }
    }
    return yi;
}

static void FillDensityMassFraction(const ablate::domain::Field& densityYiField, const std::vector<std::string>& species, const std::map<std::string, PetscReal>& yiIn, double density,
                                    std::vector<PetscReal>& conservedValues) {
    for (const auto& value : yiIn) {
        // Get the index
        auto it = std::find(species.begin(), species.end(), value.first);
        if (it != species.end()) {
            auto index = std::distance(species.begin(), it);

            conservedValues[index + densityYiField.offset] = density * value.second;
        }
    }
}

static void FillMassFraction(const std::vector<std::string>& species, const std::map<std::string, PetscReal>& yiIn, std::vector<PetscReal>& conservedValues) {
    for (const auto& value : yiIn) {
        // Get the index
        auto it = std::find(species.begin(), species.end(), value.first);
        if (it != species.end()) {
            auto index = std::distance(species.begin(), it);
            conservedValues[index] = value.second;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS create and view tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemCreateAndViewParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::string expectedViewStart;
};

class TChemCreateAndViewFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemCreateAndViewParameters> {};

TEST_P(TChemCreateAndViewFixture, ShouldCreateAndView) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    std::stringstream outputStream;

    // act
    outputStream << *eos;

    // assert the output is as expected
    auto outputString = outputStream.str();

    // only check the start because the kokkos details may change on each machine
    bool startsWith = outputString.rfind(GetParam().expectedViewStart, 0) == 0;

    ASSERT_TRUE(startsWith) << "Should start with expected string. ";
}

INSTANTIATE_TEST_SUITE_P(
    TChemTests, TChemCreateAndViewFixture,
    testing::Values((TChemCreateAndViewParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                   .thermoFile = "inputs/eos/thermo30.dat",
                                                   .expectedViewStart = "EOS: TChem\n\tmechFile: \"inputs/eos/grimech30.dat\"\n\tthermoFile: \"inputs/eos/thermo30.dat\"\n\tnumberSpecies: 53\n"},
                    (TChemCreateAndViewParameters){.mechFile = "inputs/eos/gri30.yaml", .expectedViewStart = "EOS: TChem\n\tmechFile: \"inputs/eos/gri30.yaml\"\n\tnumberSpecies: 53\n"}),
    [](const testing::TestParamInfo<TChemCreateAndViewParameters>& info) {
        return testingResources::PetscTestFixture::SanitizeTestName(info.param.mechFile.string() + "_" + info.param.thermoFile.string());
    });

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// EOS Get Species Tests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemGetSpeciesParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::vector<std::string> expectedSpecies;
};

class TChemGetSpeciesFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemGetSpeciesParameters> {};

TEST_P(TChemGetSpeciesFixture, ShouldGetCorrectSpecies) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // act
    auto species = eos->GetSpeciesVariables();

    // assert the output is as expected
    ASSERT_EQ(species, GetParam().expectedSpecies);
}

INSTANTIATE_TEST_SUITE_P(
    TChemTests, TChemGetSpeciesFixture,
    testing::Values((TChemGetSpeciesParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                .expectedSpecies = {"H2",    "H",     "O",    "O2",   "OH",    "H2O",  "HO2",   "H2O2", "C",      "CH",     "CH2",  "CH2(S)", "CH3",  "CH4",
                                                                    "CO",    "CO2",   "HCO",  "CH2O", "CH2OH", "CH3O", "CH3OH", "C2H",  "C2H2",   "C2H3",   "C2H4", "C2H5",   "C2H6", "HCCO",
                                                                    "CH2CO", "HCCOH", "N",    "NH",   "NH2",   "NH3",  "NNH",   "NO",   "NO2",    "N2O",    "HNO",  "CN",     "HCN",  "H2CN",
                                                                    "HCNN",  "HCNO",  "HOCN", "HNCO", "NCO",   "AR",   "C3H7",  "C3H8", "CH2CHO", "CH3CHO", "N2"}},
                    (TChemGetSpeciesParameters){
                        .mechFile = "inputs/eos/gri30.yaml",
                        .expectedSpecies = {"H2",    "H",    "O",     "O2",  "OH",   "H2O",  "HO2",  "H2O2", "C",    "CH",   "CH2",   "CH2(S)", "CH3",  "CH4",  "CO",     "CO2",    "HCO", "CH2O",
                                            "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2", "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH",  "N",    "NH",   "NH2",    "NH3",    "NNH", "NO",
                                            "NO2",   "N2O",  "HNO",   "CN",  "HCN",  "H2CN", "HCNN", "HCNO", "HOCN", "HNCO", "NCO",   "AR",     "C3H7", "C3H8", "CH2CHO", "CH3CHO", "N2"}}

                    ),
    [](const testing::TestParamInfo<TChemGetSpeciesParameters>& info) {
        return testingResources::PetscTestFixture::SanitizeTestName(info.param.mechFile.string() + "_" + info.param.thermoFile.string());
    });

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// EOS Thermodynamic property tests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TCTestParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::vector<ablate::domain::Field> fields;
    std::vector<PetscReal> conservedEulerValues;
    std::map<std::string, PetscReal> yiMap;
    std::optional<PetscReal> expectedTemperature;
    std::map<ablate::eos::ThermodynamicProperty, std::vector<PetscReal>> testProperties;
    PetscReal errorTolerance = 1E-5;
};

class TCThermodynamicPropertyTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TCTestParameters> {};

TEST_P(TCThermodynamicPropertyTestFixture, ShouldComputeProperty) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // combine and build the total conserved values
    auto conservedValuesSize = std::accumulate(params.fields.begin(), params.fields.end(), 0, [](int a, const ablate::domain::Field& field) { return a + field.numberComponents; });
    std::vector<PetscReal> conservedValues(conservedValuesSize + 10, 0.0); /* 10 provides some extra buffer for placement testing*/
    std::copy(params.conservedEulerValues.begin(), params.conservedEulerValues.end(), conservedValues.begin() + std::find_if(params.fields.begin(), params.fields.end(), [](const auto& field) {
                                                                                                                    return field.name == "euler";
                                                                                                                })->offset);
    FillDensityMassFraction(*std::find_if(params.fields.begin(), params.fields.end(), [](const auto& field) { return field.name == "densityYi"; }),
                            eos->GetSpeciesVariables(),
                            params.yiMap,
                            params.conservedEulerValues[0],
                            conservedValues);

    // compute the reference temperature for other calculations
    auto temperatureFunction = eos->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, params.fields);
    PetscReal computedTemperature;
    ASSERT_EQ(0, temperatureFunction.function(conservedValues.data(), &computedTemperature, temperatureFunction.context.get()));
    ASSERT_EQ(1, temperatureFunction.propertySize) << "The temperature property size should be 1";

    if (params.expectedTemperature) {
        ASSERT_LT(PetscAbs(computedTemperature - params.expectedTemperature.value()) / params.expectedTemperature.value(), 1E-5)
            << "The percent difference for computed temperature (" << params.expectedTemperature.value() << " vs " << computedTemperature << ") should be small";
    }

    // Check each of the provided property
    for (const auto& [thermodynamicProperty, expectedValue] : params.testProperties) {
        // act/assert check for compute without temperature
        auto thermodynamicFunction = eos->GetThermodynamicFunction(thermodynamicProperty, params.fields);
        std::vector<PetscReal> computedProperty(expectedValue.size(), NAN);
        ASSERT_EQ(0, thermodynamicFunction.function(conservedValues.data(), computedProperty.data(), thermodynamicFunction.context.get()));
        ASSERT_EQ(expectedValue.size(), thermodynamicFunction.propertySize) << "The " << thermodynamicProperty << " property size should be " << expectedValue.size();

        for (std::size_t c = 0; c < expectedValue.size(); c++) {
            // perform a difference check is the expectedValue is zero
            if (expectedValue[c] == 0) {
                ASSERT_LT(expectedValue[c], params.errorTolerance) << "The value for the direct function of " << to_string(thermodynamicProperty) << " (" << expectedValue[c] << " vs "
                                                                   << computedProperty[c] << ") should be near zero";
            } else {
                ASSERT_LT(PetscAbs((expectedValue[c] - computedProperty[c]) / (expectedValue[c] + 1E-30)), params.errorTolerance)
                    << "The percent difference for the direct function of " << to_string(thermodynamicProperty) << " (" << expectedValue[c] << " vs " << computedProperty[c] << ") should be small";
            }
        }

        auto thermodynamicTemperatureFunction = eos->GetThermodynamicTemperatureFunction(thermodynamicProperty, params.fields);
        computedProperty = std::vector<PetscReal>(expectedValue.size(), NAN);
        ASSERT_EQ(0, thermodynamicTemperatureFunction.function(conservedValues.data(), computedTemperature, computedProperty.data(), thermodynamicTemperatureFunction.context.get()));
        for (std::size_t c = 0; c < expectedValue.size(); c++) {
            if (expectedValue[c] == 0) {
                ASSERT_LT(expectedValue[c], params.errorTolerance) << "The value for the temperature function of " << to_string(thermodynamicProperty) << " (" << expectedValue[c] << " vs "
                                                                   << computedProperty[c] << ") should be near zero";
            } else {
                ASSERT_LT(PetscAbs((expectedValue[c] - computedProperty[c]) / (expectedValue[c] + 1E-30)), params.errorTolerance)
                    << "The percent difference for the temperature function of " << to_string(thermodynamicProperty) << " (" << expectedValue[c] << " vs " << computedProperty[c]
                    << ") should be small";
            }
        }
    }
}

TEST_P(TCThermodynamicPropertyTestFixture, ShouldComputePropertyUsingMassFraction) {
    // arrange
    std::shared_ptr<ablate::eos::TChem> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // Get the euler size
    auto eulerField = std::find_if(params.fields.begin(), params.fields.end(), [](const auto& field) { return field.name == "euler"; });

    // build the total conserved values
    auto conservedValuesSize = std::accumulate(params.fields.begin(), params.fields.end(), 0, [](int a, const ablate::domain::Field& field) { return a + field.numberComponents; });
    std::vector<PetscReal> conservedValues(conservedValuesSize + 10, 0.0); /* 10 provides some extra buffer for placement testing*/
    std::copy(params.conservedEulerValues.begin(), params.conservedEulerValues.end(), conservedValues.begin() + std::find_if(params.fields.begin(), params.fields.end(), [](const auto& field) {
                                                                                                                    return field.name == "euler";
                                                                                                                })->offset);
    // Size and fill species
    std::vector<PetscReal> yi(eos->GetSpeciesVariables().size(), 0.0);
    FillMassFraction(eos->GetSpeciesVariables(), params.yiMap, yi);

    // copy remove densityYi from the list in the params
    std::vector<ablate::domain::Field> fields;
    std::copy_if(params.fields.begin(), params.fields.end(), std::back_inserter(fields), [](const auto& field) { return field.name != "densityYi"; });

    // compute the reference temperature for other calculations
    auto temperatureFunction = eos->GetThermodynamicMassFractionFunction(ablate::eos::ThermodynamicProperty::Temperature, fields);
    PetscReal computedTemperature;
    ASSERT_EQ(0, temperatureFunction.function(conservedValues.data(), yi.data(), &computedTemperature, temperatureFunction.context.get()));

    if (params.expectedTemperature) {
        ASSERT_LT(PetscAbs(computedTemperature - params.expectedTemperature.value()) / params.expectedTemperature.value(), 1E-5)
            << "The percent difference for computed temperature (" << params.expectedTemperature.value() << " vs " << computedTemperature << ") should be small";
    }

    // Check each of the provided property
    for (const auto& [thermodynamicProperty, expectedValue] : params.testProperties) {
        // act/assert check for compute without temperature
        auto thermodynamicFunction = eos->GetThermodynamicMassFractionFunction(thermodynamicProperty, fields);
        std::vector<PetscReal> computedProperty(expectedValue.size(), NAN);
        ASSERT_EQ(0, thermodynamicFunction.function(conservedValues.data(), yi.data(), computedProperty.data(), thermodynamicFunction.context.get()));
        for (std::size_t c = 0; c < expectedValue.size(); c++) {
            // perform a difference check is the expectedValue is zero
            if (expectedValue[c] == 0) {
                ASSERT_LT(expectedValue[c], params.errorTolerance) << "The value for the direct function of " << to_string(thermodynamicProperty) << " (" << expectedValue[c] << " vs "
                                                                   << computedProperty[c] << ") should be near zero";
            } else {
                ASSERT_LT(PetscAbs((expectedValue[c] - computedProperty[c]) / (expectedValue[c] + 1E-30)), params.errorTolerance)
                    << "The percent difference for the direct function of " << to_string(thermodynamicProperty) << " (" << expectedValue[c] << " vs " << computedProperty[c] << ") should be small";
            }
        }

        auto thermodynamicTemperatureFunction = eos->GetThermodynamicTemperatureMassFractionFunction(thermodynamicProperty, fields);
        computedProperty = std::vector<PetscReal>(expectedValue.size(), NAN);
        ASSERT_EQ(0, thermodynamicTemperatureFunction.function(conservedValues.data(), yi.data(), computedTemperature, computedProperty.data(), thermodynamicTemperatureFunction.context.get()));
        for (std::size_t c = 0; c < expectedValue.size(); c++) {
            if (expectedValue[c] == 0) {
                ASSERT_LT(expectedValue[c], params.errorTolerance) << "The value for the temperature function of " << to_string(thermodynamicProperty) << " (" << expectedValue[c] << " vs "
                                                                   << computedProperty[c] << ") should be near zero";
            } else {
                ASSERT_LT(PetscAbs((expectedValue[c] - computedProperty[c]) / (expectedValue[c] + 1E-30)), params.errorTolerance)
                    << "The percent difference for the temperature function of " << to_string(thermodynamicProperty) << " (" << expectedValue[c] << " vs " << computedProperty[c]
                    << ") should be small";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    TChemTests, TCThermodynamicPropertyTestFixture,
    testing::Values(
        (TCTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                           .thermoFile = "inputs/eos/thermo30.dat",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 5}},
                           .conservedEulerValues = {1.2, 1.2 * 1.0E+05, 1.2 * 10, -1.2 * 20, 1.2 * 30},
                           .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                           .expectedTemperature = 499.2577,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {499.2577}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {99300.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {197710.5}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {264057.52}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {464.33}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1399.301411}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {1069.297887}},
                                              {ablate::eos::ThermodynamicProperty::Density, {1.2}}}},
        (TCTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                           .thermoFile = "inputs/eos/thermo30.dat",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 2}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 7}},
                           .conservedEulerValues = {1.2, 1.2 * 1.0E+05, 1.2 * 10, -1.2 * 20, 1.2 * 30},
                           .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                           .expectedTemperature = 499.2577,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {499.2577}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {99300.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {197710.5}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {264057.52}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {464.33}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1399.301411}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {1069.297887}},
                                              {ablate::eos::ThermodynamicProperty::Density, {1.2}}}},
        (TCTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                           .thermoFile = "inputs/eos/thermo30.dat",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 53}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 0}},
                           .conservedEulerValues = {1.2, 1.2 * 1.0E+05, 1.2 * 10, -1.2 * 20, 1.2 * 30},
                           .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                           .expectedTemperature = 499.2577,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {499.2577}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {99300.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {197710.5}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {264057.52}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {464.33}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1399.301411}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {1069.297887}},
                                              {ablate::eos::ThermodynamicProperty::Density, {1.2}}}},
        (TCTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                           .thermoFile = "inputs/eos/thermo30.dat",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 53}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 0}},
                           .conservedEulerValues = {0.8, 0.8 * 3.2E5, 0.0, 0.0, 0.0},
                           .yiMap = {{"O2", .3}, {"N2", .4}, {"CH2", .1}, {"NO", .2}},
                           .expectedTemperature = 762.664,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {762.664}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {320000.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {189973.54}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {557466.2}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {560.8365}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1270.738292}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {959.3732847}},
                                              {ablate::eos::ThermodynamicProperty::Density, {0.8}}}},
        (TCTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                           .thermoFile = "inputs/eos/thermo30.dat",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 55}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 2}},
                           .conservedEulerValues = {3.3, 3.3 * 1000, 0.0, 3.3 * 2, 3.3 * 4},
                           .yiMap = {{"N2", 1.0}},
                           .expectedTemperature = 418.079,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {418.079}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {990.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {409488.10}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {125077.36}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {416.04}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1048.3886}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {751.5853}},
                                              {ablate::eos::ThermodynamicProperty::Density, {3.3}}}},
        (TCTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                           .thermoFile = "inputs/eos/thermo30.dat",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 55}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 2}},
                           .conservedEulerValues = {0.01, 0.01 * 1E5, .01 * -1, .01 * -2, .01 * -3},
                           .yiMap = {{"H2", .35}, {"H2O", .35}, {"N2", .3}},
                           .expectedTemperature = 437.465757768,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {437.465757768}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {99993.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {7411.11}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {841104.3210298242}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {1013.72}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {6076.13}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {4382.03}},
                                              {ablate::eos::ThermodynamicProperty::Density, {0.01}}}},
        (TCTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                           .thermoFile = "inputs/eos/thermo30.dat",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 5}},
                           .conservedEulerValues = {999.9, 999.9 * 1.0E+04, 999.9 * -10, 999.9 * -20, 999.9 * -300},
                           .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                           .expectedTemperature = 394.59,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {394.59}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {-35250.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {281125963.5}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {245904.07895}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {623.94}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {2564.85}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {1852.33}},
                                              {ablate::eos::ThermodynamicProperty::Density, {999.9}}}},
        (TCTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                           .thermoFile = "inputs/eos/thermo30.dat",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 3}},
                           .conservedEulerValues = {1.0, 1.0 * -88491.929819300756, 0.0},
                           .yiMap = {{"N2", 1.0}},
                           .expectedTemperature = 298.15,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}}},

        (TCTestParameters){
            .mechFile = "inputs/eos/grimech30.dat",
            .thermoFile = "inputs/eos/thermo30.dat",
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 3}},
            .conservedEulerValues = {1.0, 1.0 * -49959.406703496876, 0.0},
            .yiMap = {{"N2", 1.0}},
            .expectedTemperature = 350.0,
            .testProperties = {{ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
                                {7.460590e+05, 1.069273e+06, 7.055272e+04, 4.785883e+04, 9.080069e+04, 9.707818e+04, 5.566472e+04, 6.614152e+04, 8.991894e+04, 1.162188e+05, 1.308620e+05,
                                 1.257292e+05, 1.353844e+05, 1.188149e+05, 5.401828e+04, 4.507353e+04, 6.258466e+04, 6.258237e+04, 8.324871e+04, 6.636699e+04, 7.404635e+04, 8.828951e+04,
                                 9.116586e+04, 8.561139e+04, 8.372083e+04, 9.475268e+04, 9.592900e+04, 6.316233e+04, 6.639892e+04, 7.465792e+04, 7.694628e+04, 1.007995e+05, 1.101965e+05,
                                 1.106852e+05, 6.281819e+04, 5.155943e+04, 4.282066e+04, 4.682740e+04, 5.720448e+04, 5.820967e+04, 7.055767e+04, 7.307891e+04, 6.404595e+04, 5.831064e+04,
                                 5.708868e+04, 5.750638e+04, 5.076174e+04, 2.697916e+04, 9.164036e+04, 9.281967e+04, 6.885935e+04, 6.825274e+04, 5.392193e+04}}},
            .errorTolerance = 1E-3},

        (TCTestParameters){
            .mechFile = "inputs/eos/grimech30.dat",
            .thermoFile = "inputs/eos/thermo30.dat",
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 3}},
            .conservedEulerValues = {1.0, 1.0 * 2419837.7937912419, 0.0},
            .yiMap = {{"N2", 1.0}},
            .expectedTemperature = 3000.0,
            .testProperties = {{ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
                                {4.401447e+07, 5.571873e+07, 3.536189e+06, 3.066045e+06, 5.280429e+06, 7.086383e+06, 4.236784e+06, 5.394231e+06, 4.719917e+06, 7.461729e+06, 9.382657e+06,
                                 9.338441e+06, 1.186369e+07, 1.461963e+07, 3.338867e+06, 3.472244e+06, 4.745156e+06, 6.099301e+06, 7.376036e+06, 7.489415e+06, 8.456529e+06, 6.201523e+06,
                                 7.700047e+06, 8.726732e+06, 1.006060e+07, 1.118903e+07, 1.239586e+07, 4.817136e+06, 5.843981e+06, 6.082402e+06, 4.013085e+06, 6.105222e+06, 8.300475e+06,
                                 1.027588e+07, 4.726630e+06, 3.168311e+06, 3.227706e+06, 3.530461e+06, 4.750059e+06, 3.741030e+06, 5.402331e+06, 6.678596e+06, 4.877817e+06, 4.658460e+06,
                                 4.383264e+06, 4.569753e+06, 3.683059e+06, 1.405856e+06, 1.109638e+07, 1.193612e+07, 6.659771e+06, 7.583124e+06, 3.310248e+06}}},
            .errorTolerance = 1E-3},
        ///////// with yaml input ///////
        (TCTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 5}},
                           .conservedEulerValues = {1.2, 1.2 * 1.0E+05, 1.2 * 10, -1.2 * 20, 1.2 * 30},
                           .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                           .expectedTemperature = 499.2577,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {499.2577}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {99300.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {197710.5}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {264057.52}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {464.33}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1399.301411}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {1069.297887}},
                                              {ablate::eos::ThermodynamicProperty::Density, {1.2}}}},
        (TCTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 2}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 7}},
                           .conservedEulerValues = {1.2, 1.2 * 1.0E+05, 1.2 * 10, -1.2 * 20, 1.2 * 30},
                           .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                           .expectedTemperature = 499.2577,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {499.2577}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {99300.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {197710.5}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {264057.52}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {464.33}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1399.301411}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {1069.297887}},
                                              {ablate::eos::ThermodynamicProperty::Density, {1.2}}}},
        (TCTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 53}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 0}},
                           .conservedEulerValues = {1.2, 1.2 * 1.0E+05, 1.2 * 10, -1.2 * 20, 1.2 * 30},
                           .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                           .expectedTemperature = 499.2577,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {499.2577}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {99300.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {197710.5}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {264057.52}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {464.33}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1399.301411}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {1069.297887}},
                                              {ablate::eos::ThermodynamicProperty::Density, {1.2}}}},
        (TCTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 53}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 0}},
                           .conservedEulerValues = {0.8, 0.8 * 3.2E5, 0.0, 0.0, 0.0},
                           .yiMap = {{"O2", .3}, {"N2", .4}, {"CH2", .1}, {"NO", .2}},
                           .expectedTemperature = 762.664,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {762.664}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {320000.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {189973.54}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {557466.2}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {560.8365}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1270.738292}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {959.3732847}},
                                              {ablate::eos::ThermodynamicProperty::Density, {0.8}}}},
        (TCTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 55}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 2}},
                           .conservedEulerValues = {3.3, 3.3 * 1000, 0.0, 3.3 * 2, 3.3 * 4},
                           .yiMap = {{"N2", 1.0}},
                           .expectedTemperature = 418.079,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {418.079}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {990.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {409488.10}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {125077.36}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {416.04}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {1048.3886}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {751.5853}},
                                              {ablate::eos::ThermodynamicProperty::Density, {3.3}}}},
        (TCTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 55}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 2}},
                           .conservedEulerValues = {0.01, 0.01 * 1E5, .01 * -1, .01 * -2, .01 * -3},
                           .yiMap = {{"H2", .35}, {"H2O", .35}, {"N2", .3}},
                           .expectedTemperature = 437.465757768,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {437.465757768}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {99993.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {7411.11}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {841104.3210298242}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {1013.72}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {6076.13}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {4382.03}},
                                              {ablate::eos::ThermodynamicProperty::Density, {0.01}}}},
        (TCTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 5}},
                           .conservedEulerValues = {999.9, 999.9 * 1.0E+04, 999.9 * -10, 999.9 * -20, 999.9 * -300},
                           .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                           .expectedTemperature = 394.59,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::Temperature, {394.59}},
                                              {ablate::eos::ThermodynamicProperty::InternalSensibleEnergy, {-35250.0}},
                                              {ablate::eos::ThermodynamicProperty::Pressure, {281125963.5}},
                                              {ablate::eos::ThermodynamicProperty::SensibleEnthalpy, {245904.07895}},
                                              {ablate::eos::ThermodynamicProperty::SpeedOfSound, {623.94}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, {2564.85}},
                                              {ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, {1852.33}},
                                              {ablate::eos::ThermodynamicProperty::Density, {999.9}}}},
        (TCTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 3}},
                           .conservedEulerValues = {1.0, 1.0 * -88491.929819300756, 0.0},
                           .yiMap = {{"N2", 1.0}},
                           .expectedTemperature = 298.15,
                           .testProperties = {{ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
                                               {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}}},

        (TCTestParameters){
            .mechFile = "inputs/eos/gri30.yaml",
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 3}},
            .conservedEulerValues = {1.0, 1.0 * -49959.406703496876, 0.0},
            .yiMap = {{"N2", 1.0}},
            .expectedTemperature = 350.0,
            .testProperties = {{ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
                                {7.460590e+05, 1.069273e+06, 7.055272e+04, 4.785883e+04, 9.080069e+04, 9.707818e+04, 5.566472e+04, 6.614152e+04, 8.991894e+04, 1.162188e+05, 1.308620e+05,
                                 1.257292e+05, 1.353844e+05, 1.188149e+05, 5.401828e+04, 4.507353e+04, 6.258466e+04, 6.258237e+04, 8.324871e+04, 6.636699e+04, 7.404635e+04, 8.828951e+04,
                                 9.116586e+04, 8.561139e+04, 8.372083e+04, 9.475268e+04, 9.592900e+04, 6.316233e+04, 6.639892e+04, 7.465792e+04, 7.694628e+04, 1.007995e+05, 1.101965e+05,
                                 1.106852e+05, 6.281819e+04, 5.155943e+04, 4.282066e+04, 4.682740e+04, 5.720448e+04, 5.820967e+04, 7.055767e+04, 7.307891e+04, 6.404595e+04, 5.831064e+04,
                                 5.708868e+04, 5.750638e+04, 5.076174e+04, 2.697916e+04, 9.164036e+04, 9.281967e+04, 6.885935e+04, 6.825274e+04, 5.392193e+04}}},
            .errorTolerance = 1E-3},

        (TCTestParameters){
            .mechFile = "inputs/eos/gri30.yaml",
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 0}, ablate::domain::Field{.name = "densityYi", .numberComponents = 53, .offset = 3}},
            .conservedEulerValues = {1.0, 1.0 * 2419837.7937912419, 0.0},
            .yiMap = {{"N2", 1.0}},
            .expectedTemperature = 3000.0,
            .testProperties = {{ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
                                {4.401447e+07, 5.571873e+07, 3.536189e+06, 3.066045e+06, 5.280429e+06, 7.086383e+06, 4.236784e+06, 5.394231e+06, 4.719917e+06, 7.461729e+06, 9.382657e+06,
                                 9.338441e+06, 1.186369e+07, 1.461963e+07, 3.338867e+06, 3.472244e+06, 4.745156e+06, 6.099301e+06, 7.376036e+06, 7.489415e+06, 8.456529e+06, 6.201523e+06,
                                 7.700047e+06, 8.726732e+06, 1.006060e+07, 1.118903e+07, 1.239586e+07, 4.817136e+06, 5.843981e+06, 6.082402e+06, 4.013085e+06, 6.105222e+06, 8.300475e+06,
                                 1.027588e+07, 4.726630e+06, 3.168311e+06, 3.227706e+06, 3.530461e+06, 4.750059e+06, 3.741030e+06, 5.402331e+06, 6.678596e+06, 4.877817e+06, 4.658460e+06,
                                 4.383264e+06, 4.569753e+06, 3.683059e+06, 1.405856e+06, 1.109638e+07, 1.193612e+07, 6.659771e+06, 7.583124e+06, 3.310248e+06}}},
            .errorTolerance = 1E-3}

        ),

    [](const testing::TestParamInfo<TCTestParameters>& info) { return "case_" + std::to_string(info.index) + "_" + info.param.mechFile.stem().string(); });

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// Perfect Gas FieldFunctionTests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemFieldFunctionTestParameters {
    // eos init
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;

    // field function init
    ablate::eos::ThermodynamicProperty property1;
    ablate::eos::ThermodynamicProperty property2;

    // inputs
    PetscReal property1Value;
    PetscReal property2Value;
    std::vector<PetscReal> velocity;
    std::map<std::string, PetscReal> yiMap;
    std::vector<PetscReal> expectedEulerValue;
};

class TChemFieldFunctionTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemFieldFunctionTestParameters> {};

TEST_P(TChemFieldFunctionTestFixture, ShouldComputeField) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);
    auto yi = GetMassFraction(eos->GetSpeciesVariables(), GetParam().yiMap);

    // get the test params
    const auto& params = GetParam();
    std::vector<PetscReal> actualEulerValue(params.expectedEulerValue.size(), NAN);
    std::vector<PetscReal> actualDensityYiValue(eos->GetSpeciesVariables().size(), NAN);

    // act
    auto stateEulerFunction = eos->GetFieldFunctionFunction("euler", params.property1, params.property2, {ablate::eos::EOS::YI});
    stateEulerFunction(params.property1Value, params.property2Value, (PetscInt)params.velocity.size(), params.velocity.data(), yi.data(), actualEulerValue.data());
    auto stateDensityYiFunction = eos->GetFieldFunctionFunction("densityYi", params.property1, params.property2, {ablate::eos::EOS::YI});
    stateDensityYiFunction(params.property1Value, params.property2Value, (PetscInt)params.velocity.size(), params.velocity.data(), yi.data(), actualDensityYiValue.data());

    // assert
    for (std::size_t c = 0; c < params.expectedEulerValue.size(); c++) {
        ASSERT_LT(PetscAbs(params.expectedEulerValue[c] - actualEulerValue[c]) / (params.expectedEulerValue[c] + 1E-30), 1E-3)
            << "for component[" << c << "] of expectedEulerValue (" << params.expectedEulerValue[c] << " vs " << actualEulerValue[c] << ")";
    }
    for (std::size_t c = 0; c < yi.size(); c++) {
        ASSERT_LT(PetscAbs(yi[c] * params.expectedEulerValue[0] - actualDensityYiValue[c]) / (yi[c] * params.expectedEulerValue[0] + 1E-30), 1E-3)
            << "for component[" << c << "] of densityYi (" << yi[c] * params.expectedEulerValue[0] << " vs " << actualDensityYiValue[c] << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(TChemTests, TChemFieldFunctionTestFixture,
                         testing::Values((TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = 499.25,
                                                                            .property2Value = 197710.5,
                                                                            .velocity = {10, 20, 30},
                                                                            .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                            .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = 762.664,
                                                                            .property2Value = 189973.54,
                                                                            .velocity = {0.0, 0.0, 0.0},
                                                                            .yiMap = {{"O2", .3}, {"N2", .4}, {"CH2", .1}, {"NO", .2}},
                                                                            .expectedEulerValue = {0.8, 0.8 * 3.2E5, 0.0, 0.0, 0.0}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = 418.079,
                                                                            .property2Value = 409488.10,
                                                                            .velocity = {0, 2, 4},
                                                                            .yiMap = {{"N2", 1.0}},
                                                                            .expectedEulerValue = {3.3, 3.3 * 1000, 0.0, 3.3 * 2, 3.3 * 4}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property1Value = 7411.11,
                                                                            .property2Value = 437.46,
                                                                            .velocity = {-1, -2, -3},
                                                                            .yiMap = {{"H2", .35}, {"H2O", .35}, {"N2", .3}},
                                                                            .expectedEulerValue = {0.01, 0.01 * 1E5, .01 * -1, .01 * -2, .01 * -3}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property1Value = 281125963.5,
                                                                            .property2Value = 394.59,
                                                                            .velocity = {-10, -20, -300},
                                                                            .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                            .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},

                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                            .property1Value = 281125963.5,
                                                                            .property2Value = -35256.942891550425,
                                                                            .velocity = {-10, -20, -300},
                                                                            .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                            .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = -35256.942891550425,
                                                                            .property2Value = 281125963.5,
                                                                            .velocity = {-10, -20, -300},
                                                                            .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                            .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},

                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = 99291.694615827029,
                                                                            .property2Value = 197710.5,
                                                                            .velocity = {10, 20, 30},
                                                                            .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                            .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                            .thermoFile = "inputs/eos/thermo30.dat",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                            .property1Value = 197710.5,
                                                                            .property2Value = 99291.694615827029,
                                                                            .velocity = {10, 20, 30},
                                                                            .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                            .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}},
                                         ///////// with yaml input ///////
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = 499.25,
                                                                            .property2Value = 197710.5,
                                                                            .velocity = {10, 20, 30},
                                                                            .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                            .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = 762.664,
                                                                            .property2Value = 189973.54,
                                                                            .velocity = {0.0, 0.0, 0.0},
                                                                            .yiMap = {{"O2", .3}, {"N2", .4}, {"CH2", .1}, {"NO", .2}},
                                                                            .expectedEulerValue = {0.8, 0.8 * 3.2E5, 0.0, 0.0, 0.0}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = 418.079,
                                                                            .property2Value = 409488.10,
                                                                            .velocity = {0, 2, 4},
                                                                            .yiMap = {{"N2", 1.0}},
                                                                            .expectedEulerValue = {3.3, 3.3 * 1000, 0.0, 3.3 * 2, 3.3 * 4}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property1Value = 7411.11,
                                                                            .property2Value = 437.46,
                                                                            .velocity = {-1, -2, -3},
                                                                            .yiMap = {{"H2", .35}, {"H2O", .35}, {"N2", .3}},
                                                                            .expectedEulerValue = {0.01, 0.01 * 1E5, .01 * -1, .01 * -2, .01 * -3}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                            .property1Value = 281125963.5,
                                                                            .property2Value = 394.59,
                                                                            .velocity = {-10, -20, -300},
                                                                            .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                            .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},

                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                            .property1Value = 281125963.5,
                                                                            .property2Value = -35256.942891550425,
                                                                            .velocity = {-10, -20, -300},
                                                                            .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                            .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = -35256.942891550425,
                                                                            .property2Value = 281125963.5,
                                                                            .velocity = {-10, -20, -300},
                                                                            .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                            .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},

                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property1Value = 99291.694615827029,
                                                                            .property2Value = 197710.5,
                                                                            .velocity = {10, 20, 30},
                                                                            .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                            .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}},
                                         (TChemFieldFunctionTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                                            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                            .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                            .property1Value = 197710.5,
                                                                            .property2Value = 99291.694615827029,
                                                                            .velocity = {10, 20, 30},
                                                                            .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                            .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}}

                                         ),

                         [](const testing::TestParamInfo<TChemFieldFunctionTestParameters>& info) {
                             return std::to_string(info.index) + "_from_" + std::string(to_string(info.param.property1)) + "_" + std::string(to_string(info.param.property2)) + "_with_" +
                                    info.param.mechFile.stem().string();
                         });

struct TCElementTestParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::map<std::string, double> expectedElementInformation;
};

class TCElementTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TCElementTestParameters> {};

TEST_P(TCElementTestFixture, ShouldDetermineElements) {
    // arrange
    auto eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // act
    auto elementInformation = eos->GetElementInformation();

    // assert
    ASSERT_EQ(params.expectedElementInformation, elementInformation);
}

INSTANTIATE_TEST_SUITE_P(TChemTests, TCElementTestFixture,
                         testing::Values(
                             (TCElementTestParameters){
                                 .mechFile = "inputs/eos/grimech30.dat",
                                 .thermoFile = "inputs/eos/thermo30.dat",
                                 .expectedElementInformation = {{"AR", 39.948}, {"C", 12.01115}, {"H", 1.00797}, {"N", 14.0067}, {"O", 15.9994}},
                             },
                             (TCElementTestParameters){.mechFile = "inputs/eos/gri30.yaml",
                                                       .expectedElementInformation = {{"AR", 39.948}, {"C", 12.01115}, {"H", 1.00797}, {"N", 14.0067}, {"O", 15.9994}}}),
                         [](const testing::TestParamInfo<TCElementTestParameters>& info) { return TCElementTestFixture::SanitizeTestName(info.param.mechFile.string()); });

struct TCSpeciesInformationTestParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::map<std::string, double> expectedSpeciesMolecularMass;
    std::map<std::string, std::map<std::string, int>> expectedSpeciesElementInformation;
};

class TCSpeciesInformationTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TCSpeciesInformationTestParameters> {};

TEST_P(TCSpeciesInformationTestFixture, ShouldDetermineSpeciesElementInformation) {
    // arrange
    auto eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // act
    auto speciesElementInformation = eos->GetSpeciesElementalInformation();

    // assert
    ASSERT_EQ(params.expectedSpeciesElementInformation, speciesElementInformation);
}

TEST_P(TCSpeciesInformationTestFixture, ShouldDetermineSpeciesMolecularMass) {
    // arrange
    auto eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // act
    auto molecularMass = eos->GetSpeciesMolecularMass();

    // assert
    ASSERT_EQ(params.expectedSpeciesMolecularMass.size(), molecularMass.size()) << "the number of species should be correct";
    for (const auto& [species, mw] : params.expectedSpeciesMolecularMass) {
        EXPECT_NEAR(mw, molecularMass[species], 1E-2) << "the mw for " << species << " should be correct";
    }
}

INSTANTIATE_TEST_SUITE_P(
    TChemTests, TCSpeciesInformationTestFixture,
    testing::Values(
        (TCSpeciesInformationTestParameters){
            .mechFile = "inputs/eos/grimech30.dat",
            .thermoFile = "inputs/eos/thermo30.dat",
            .expectedSpeciesMolecularMass = {{"AR", 39.948},     {"C", 12.0112},    {"C2H", 25.0303},   {"C2H2", 26.0382},  {"C2H3", 27.0462},   {"C2H4", 28.0542},   {"C2H5", 29.0622},
                                             {"C2H6", 30.0701},  {"C3H7", 43.0892}, {"C3H8", 44.0972},  {"CH", 13.0191},    {"CH2", 14.0271},    {"CH2(S)", 14.0271}, {"CH2CHO", 43.0456},
                                             {"CH2CO", 42.0376}, {"CH2O", 30.0265}, {"CH2OH", 31.0345}, {"CH3", 15.0351},   {"CH3CHO", 44.0536}, {"CH3O", 31.0345},   {"CH3OH", 32.0424},
                                             {"CH4", 16.043},    {"CN", 26.0179},   {"CO", 28.0106},    {"CO2", 44.01},     {"H", 1.00797},      {"H2", 2.01594},     {"H2CN", 28.0338},
                                             {"H2O", 18.0153},   {"H2O2", 34.0147}, {"HCCO", 41.0297},  {"HCCOH", 42.0376}, {"HCN", 27.0258},    {"HCNN", 41.0325},   {"HCNO", 43.0252},
                                             {"HCO", 29.0185},   {"HNCO", 43.0252}, {"HNO", 31.0141},   {"HO2", 33.0068},   {"HOCN", 43.0252},   {"N", 14.0067},      {"N2", 28.0134},
                                             {"N2O", 44.0128},   {"NCO", 42.0173},  {"NH", 15.0147},    {"NH2", 16.0226},   {"NH3", 17.0306},    {"NNH", 29.0214},    {"NO", 30.0061},
                                             {"NO2", 46.0055},   {"O", 15.9994},    {"O2", 31.9988},    {"OH", 17.0074}},
            .expectedSpeciesElementInformation = {{"AR", {{"AR", 1}, {"C", 0}, {"H", 0}, {"N", 0}, {"O", 0}}},     {"C", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 0}, {"O", 0}}},
                                                  {"C2H", {{"AR", 0}, {"C", 2}, {"H", 1}, {"N", 0}, {"O", 0}}},    {"C2H2", {{"AR", 0}, {"C", 2}, {"H", 2}, {"N", 0}, {"O", 0}}},
                                                  {"C2H3", {{"AR", 0}, {"C", 2}, {"H", 3}, {"N", 0}, {"O", 0}}},   {"C2H4", {{"AR", 0}, {"C", 2}, {"H", 4}, {"N", 0}, {"O", 0}}},
                                                  {"C2H5", {{"AR", 0}, {"C", 2}, {"H", 5}, {"N", 0}, {"O", 0}}},   {"C2H6", {{"AR", 0}, {"C", 2}, {"H", 6}, {"N", 0}, {"O", 0}}},
                                                  {"C3H7", {{"AR", 0}, {"C", 3}, {"H", 7}, {"N", 0}, {"O", 0}}},   {"C3H8", {{"AR", 0}, {"C", 3}, {"H", 8}, {"N", 0}, {"O", 0}}},
                                                  {"CH", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 0}, {"O", 0}}},     {"CH2", {{"AR", 0}, {"C", 1}, {"H", 2}, {"N", 0}, {"O", 0}}},
                                                  {"CH2(S)", {{"AR", 0}, {"C", 1}, {"H", 2}, {"N", 0}, {"O", 0}}}, {"CH2CHO", {{"AR", 0}, {"C", 2}, {"H", 3}, {"N", 0}, {"O", 1}}},
                                                  {"CH2CO", {{"AR", 0}, {"C", 2}, {"H", 2}, {"N", 0}, {"O", 1}}},  {"CH2O", {{"AR", 0}, {"C", 1}, {"H", 2}, {"N", 0}, {"O", 1}}},
                                                  {"CH2OH", {{"AR", 0}, {"C", 1}, {"H", 3}, {"N", 0}, {"O", 1}}},  {"CH3", {{"AR", 0}, {"C", 1}, {"H", 3}, {"N", 0}, {"O", 0}}},
                                                  {"CH3CHO", {{"AR", 0}, {"C", 2}, {"H", 4}, {"N", 0}, {"O", 1}}}, {"CH3O", {{"AR", 0}, {"C", 1}, {"H", 3}, {"N", 0}, {"O", 1}}},
                                                  {"CH3OH", {{"AR", 0}, {"C", 1}, {"H", 4}, {"N", 0}, {"O", 1}}},  {"CH4", {{"AR", 0}, {"C", 1}, {"H", 4}, {"N", 0}, {"O", 0}}},
                                                  {"CN", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 1}, {"O", 0}}},     {"CO", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 0}, {"O", 1}}},
                                                  {"CO2", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 0}, {"O", 2}}},    {"H", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 0}, {"O", 0}}},
                                                  {"H2", {{"AR", 0}, {"C", 0}, {"H", 2}, {"N", 0}, {"O", 0}}},     {"H2CN", {{"AR", 0}, {"C", 1}, {"H", 2}, {"N", 1}, {"O", 0}}},
                                                  {"H2O", {{"AR", 0}, {"C", 0}, {"H", 2}, {"N", 0}, {"O", 1}}},    {"H2O2", {{"AR", 0}, {"C", 0}, {"H", 2}, {"N", 0}, {"O", 2}}},
                                                  {"HCCO", {{"AR", 0}, {"C", 2}, {"H", 1}, {"N", 0}, {"O", 1}}},   {"HCCOH", {{"AR", 0}, {"C", 2}, {"H", 2}, {"N", 0}, {"O", 1}}},
                                                  {"HCN", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 1}, {"O", 0}}},    {"HCNN", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 2}, {"O", 0}}},
                                                  {"HCNO", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 1}, {"O", 1}}},   {"HCO", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 0}, {"O", 1}}},
                                                  {"HNCO", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 1}, {"O", 1}}},   {"HNO", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 1}, {"O", 1}}},
                                                  {"HO2", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 0}, {"O", 2}}},    {"HOCN", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 1}, {"O", 1}}},
                                                  {"N", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 1}, {"O", 0}}},      {"N2", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 2}, {"O", 0}}},
                                                  {"N2O", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 2}, {"O", 1}}},    {"NCO", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 1}, {"O", 1}}},
                                                  {"NH", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 1}, {"O", 0}}},     {"NH2", {{"AR", 0}, {"C", 0}, {"H", 2}, {"N", 1}, {"O", 0}}},
                                                  {"NH3", {{"AR", 0}, {"C", 0}, {"H", 3}, {"N", 1}, {"O", 0}}},    {"NNH", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 2}, {"O", 0}}},
                                                  {"NO", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 1}, {"O", 1}}},     {"NO2", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 1}, {"O", 2}}},
                                                  {"O", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 0}, {"O", 1}}},      {"O2", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 0}, {"O", 2}}},
                                                  {"OH", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 0}, {"O", 1}}}}},
        (TCSpeciesInformationTestParameters){
            .mechFile = "inputs/eos/gri30.yaml",
            .expectedSpeciesMolecularMass = {{"AR", 39.948},     {"C", 12.0112},    {"C2H", 25.0303},   {"C2H2", 26.0382},  {"C2H3", 27.0462},   {"C2H4", 28.0542},   {"C2H5", 29.0622},
                                             {"C2H6", 30.0701},  {"C3H7", 43.0892}, {"C3H8", 44.0972},  {"CH", 13.0191},    {"CH2", 14.0271},    {"CH2(S)", 14.0271}, {"CH2CHO", 43.0456},
                                             {"CH2CO", 42.0376}, {"CH2O", 30.0265}, {"CH2OH", 31.0345}, {"CH3", 15.0351},   {"CH3CHO", 44.0536}, {"CH3O", 31.0345},   {"CH3OH", 32.0424},
                                             {"CH4", 16.043},    {"CN", 26.0179},   {"CO", 28.0106},    {"CO2", 44.01},     {"H", 1.00797},      {"H2", 2.01594},     {"H2CN", 28.0338},
                                             {"H2O", 18.0153},   {"H2O2", 34.0147}, {"HCCO", 41.0297},  {"HCCOH", 42.0376}, {"HCN", 27.0258},    {"HCNN", 41.0325},   {"HCNO", 43.0252},
                                             {"HCO", 29.0185},   {"HNCO", 43.0252}, {"HNO", 31.0141},   {"HO2", 33.0068},   {"HOCN", 43.0252},   {"N", 14.0067},      {"N2", 28.0134},
                                             {"N2O", 44.0128},   {"NCO", 42.0173},  {"NH", 15.0147},    {"NH2", 16.0226},   {"NH3", 17.0306},    {"NNH", 29.0214},    {"NO", 30.0061},
                                             {"NO2", 46.0055},   {"O", 15.9994},    {"O2", 31.9988},    {"OH", 17.0074}},
            .expectedSpeciesElementInformation = {{"AR", {{"AR", 1}, {"C", 0}, {"H", 0}, {"N", 0}, {"O", 0}}},     {"C", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 0}, {"O", 0}}},
                                                  {"C2H", {{"AR", 0}, {"C", 2}, {"H", 1}, {"N", 0}, {"O", 0}}},    {"C2H2", {{"AR", 0}, {"C", 2}, {"H", 2}, {"N", 0}, {"O", 0}}},
                                                  {"C2H3", {{"AR", 0}, {"C", 2}, {"H", 3}, {"N", 0}, {"O", 0}}},   {"C2H4", {{"AR", 0}, {"C", 2}, {"H", 4}, {"N", 0}, {"O", 0}}},
                                                  {"C2H5", {{"AR", 0}, {"C", 2}, {"H", 5}, {"N", 0}, {"O", 0}}},   {"C2H6", {{"AR", 0}, {"C", 2}, {"H", 6}, {"N", 0}, {"O", 0}}},
                                                  {"C3H7", {{"AR", 0}, {"C", 3}, {"H", 7}, {"N", 0}, {"O", 0}}},   {"C3H8", {{"AR", 0}, {"C", 3}, {"H", 8}, {"N", 0}, {"O", 0}}},
                                                  {"CH", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 0}, {"O", 0}}},     {"CH2", {{"AR", 0}, {"C", 1}, {"H", 2}, {"N", 0}, {"O", 0}}},
                                                  {"CH2(S)", {{"AR", 0}, {"C", 1}, {"H", 2}, {"N", 0}, {"O", 0}}}, {"CH2CHO", {{"AR", 0}, {"C", 2}, {"H", 3}, {"N", 0}, {"O", 1}}},
                                                  {"CH2CO", {{"AR", 0}, {"C", 2}, {"H", 2}, {"N", 0}, {"O", 1}}},  {"CH2O", {{"AR", 0}, {"C", 1}, {"H", 2}, {"N", 0}, {"O", 1}}},
                                                  {"CH2OH", {{"AR", 0}, {"C", 1}, {"H", 3}, {"N", 0}, {"O", 1}}},  {"CH3", {{"AR", 0}, {"C", 1}, {"H", 3}, {"N", 0}, {"O", 0}}},
                                                  {"CH3CHO", {{"AR", 0}, {"C", 2}, {"H", 4}, {"N", 0}, {"O", 1}}}, {"CH3O", {{"AR", 0}, {"C", 1}, {"H", 3}, {"N", 0}, {"O", 1}}},
                                                  {"CH3OH", {{"AR", 0}, {"C", 1}, {"H", 4}, {"N", 0}, {"O", 1}}},  {"CH4", {{"AR", 0}, {"C", 1}, {"H", 4}, {"N", 0}, {"O", 0}}},
                                                  {"CN", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 1}, {"O", 0}}},     {"CO", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 0}, {"O", 1}}},
                                                  {"CO2", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 0}, {"O", 2}}},    {"H", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 0}, {"O", 0}}},
                                                  {"H2", {{"AR", 0}, {"C", 0}, {"H", 2}, {"N", 0}, {"O", 0}}},     {"H2CN", {{"AR", 0}, {"C", 1}, {"H", 2}, {"N", 1}, {"O", 0}}},
                                                  {"H2O", {{"AR", 0}, {"C", 0}, {"H", 2}, {"N", 0}, {"O", 1}}},    {"H2O2", {{"AR", 0}, {"C", 0}, {"H", 2}, {"N", 0}, {"O", 2}}},
                                                  {"HCCO", {{"AR", 0}, {"C", 2}, {"H", 1}, {"N", 0}, {"O", 1}}},   {"HCCOH", {{"AR", 0}, {"C", 2}, {"H", 2}, {"N", 0}, {"O", 1}}},
                                                  {"HCN", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 1}, {"O", 0}}},    {"HCNN", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 2}, {"O", 0}}},
                                                  {"HCNO", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 1}, {"O", 1}}},   {"HCO", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 0}, {"O", 1}}},
                                                  {"HNCO", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 1}, {"O", 1}}},   {"HNO", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 1}, {"O", 1}}},
                                                  {"HO2", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 0}, {"O", 2}}},    {"HOCN", {{"AR", 0}, {"C", 1}, {"H", 1}, {"N", 1}, {"O", 1}}},
                                                  {"N", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 1}, {"O", 0}}},      {"N2", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 2}, {"O", 0}}},
                                                  {"N2O", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 2}, {"O", 1}}},    {"NCO", {{"AR", 0}, {"C", 1}, {"H", 0}, {"N", 1}, {"O", 1}}},
                                                  {"NH", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 1}, {"O", 0}}},     {"NH2", {{"AR", 0}, {"C", 0}, {"H", 2}, {"N", 1}, {"O", 0}}},
                                                  {"NH3", {{"AR", 0}, {"C", 0}, {"H", 3}, {"N", 1}, {"O", 0}}},    {"NNH", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 2}, {"O", 0}}},
                                                  {"NO", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 1}, {"O", 1}}},     {"NO2", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 1}, {"O", 2}}},
                                                  {"O", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 0}, {"O", 1}}},      {"O2", {{"AR", 0}, {"C", 0}, {"H", 0}, {"N", 0}, {"O", 2}}},
                                                  {"OH", {{"AR", 0}, {"C", 0}, {"H", 1}, {"N", 0}, {"O", 1}}}}}),
    [](const testing::TestParamInfo<TCSpeciesInformationTestParameters>& info) { return TCElementTestFixture::SanitizeTestName(info.param.mechFile.string()); });

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// EOS Thermodynamic property tests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TCComputeSourceTestParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    PetscReal dt;
    std::vector<PetscReal> inputEulerValues;
    std::vector<PetscReal> inputDensityYiValues;

    std::vector<PetscReal> expectedEulerSource;
    std::vector<PetscReal> expectedDensityYiSource;

    PetscReal errorTolerance = 1E-3;
};

class TCComputeSourceTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TCComputeSourceTestParameters> {};

TEST_P(TCComputeSourceTestFixture, ShouldComputeCorrectSource) {
    // ARRANGE
    auto eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // create a zeroD domain for testing
    auto domain = std::make_shared<ablate::domain::BoxMesh>("zeroD",
                                                            std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>>{std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)},
                                                            std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                                            std::vector<int>{1},
                                                            std::vector<double>{0.0},
                                                            std::vector<double>{1.0});
    domain->InitializeSubDomains();

    // get the test params
    const auto& params = GetParam();

    // copy over the initial euler values
    PetscScalar* solution;
    VecGetArray(domain->GetSolutionVector(), &solution) >> ablate::utilities::PetscUtilities::checkError;
    PetscScalar* eulerField = nullptr;
    DMPlexPointLocalFieldRef(domain->GetDM(), 0, domain->GetField("euler").id, solution, &eulerField) >> ablate::utilities::PetscUtilities::checkError;
    // copy over euler
    for (std::size_t i = 0; i < GetParam().inputEulerValues.size(); i++) {
        eulerField[i] = GetParam().inputEulerValues[i];
    }

    // copy over the initial densityYi values
    PetscScalar* densityYiField = nullptr;
    DMPlexPointLocalFieldRef(domain->GetDM(), 0, domain->GetField("densityYi").id, solution, &densityYiField) >> ablate::utilities::PetscUtilities::checkError;
    // copy over euler
    for (std::size_t i = 0; i < GetParam().inputDensityYiValues.size(); i++) {
        densityYiField[i] = GetParam().inputDensityYiValues[i];
    }
    VecRestoreArray(domain->GetSolutionVector(), &solution) >> ablate::utilities::PetscUtilities::checkError;

    // create a copy to store f calculation
    Vec computedF;
    DMGetLocalVector(domain->GetDM(), &computedF) >> ablate::utilities::PetscUtilities::checkError;
    VecZeroEntries(computedF) >> ablate::utilities::PetscUtilities::checkError;

    // ACT
    ablate::solver::DynamicRange range;
    range.Add(0);
    auto sourceTermCalculator = eos->CreateSourceCalculator(domain->GetFields(), range.GetRange());

    // Perform prestep
    sourceTermCalculator->ComputeSource(range.GetRange(), 0.0, GetParam().dt, domain->GetSolutionVector());

    // perform source add
    sourceTermCalculator->AddSource(range.GetRange(), domain->GetSolutionVector(), computedF);

    // ASSERT
    PetscScalar* sourceArray;
    VecGetArray(computedF, &sourceArray) >> ablate::utilities::PetscUtilities::checkError;
    PetscScalar* eulerSource = nullptr;
    DMPlexPointLocalFieldRef(domain->GetDM(), 0, domain->GetField("euler").id, sourceArray, &eulerSource) >> ablate::utilities::PetscUtilities::checkError;
    for (std::size_t c = 0; c < GetParam().expectedEulerSource.size(); c++) {
        if (PetscAbs(GetParam().expectedEulerSource[c]) == 0) {
            ASSERT_LT(PetscAbs(eulerSource[c]), params.errorTolerance) << "The computed value of source for index " << c << " is " << eulerSource[c] << "), it should be near zero";
        } else {
            ASSERT_LT(PetscAbs((GetParam().expectedEulerSource[c] - eulerSource[c]) / (GetParam().expectedEulerSource[c] + 1E-30)), params.errorTolerance)
                << "The percent difference for the expected and actual source (" << GetParam().expectedEulerSource[c] << " vs " << eulerSource[c] << ") should be small for index " << c;
        }
    }
    PetscScalar* densityYiSource = nullptr;
    DMPlexPointLocalFieldRef(domain->GetDM(), 0, domain->GetField("densityYi").id, sourceArray, &densityYiSource) >> ablate::utilities::PetscUtilities::checkError;
    for (std::size_t c = 0; c < GetParam().expectedDensityYiSource.size(); c++) {
        if (PetscAbs(GetParam().expectedDensityYiSource[c]) < params.errorTolerance) {
            ASSERT_LT(PetscAbs(densityYiSource[c]), params.errorTolerance) << "The computed value of source for index " << c << " is " << densityYiSource[c] << "), it should be near zero";
        } else {
            ASSERT_LT(PetscAbs((GetParam().expectedDensityYiSource[c] - densityYiSource[c]) / (GetParam().expectedDensityYiSource[c] + 1E-30)), params.errorTolerance)
                << "The percent difference for the expected and actual source (" << GetParam().expectedDensityYiSource[c] << " vs " << densityYiSource[c] << ") should be small for index " << c;
        }
    }
    VecRestoreArray(computedF, &sourceArray) >> ablate::utilities::PetscUtilities::checkError;

    DMRestoreLocalVector(domain->GetDM(), &computedF) >> ablate::utilities::PetscUtilities::checkError;
}

INSTANTIATE_TEST_SUITE_P(
    TChemTests, TCComputeSourceTestFixture,
    testing::Values(
        (TCComputeSourceTestParameters){
            .mechFile = "inputs/eos/grimech30.dat",
            .thermoFile = "inputs/eos/thermo30.dat",
            .dt = 0.0001,
            .inputEulerValues = {0.280629, 212565., 0.},
            .inputDensityYiValues = {0., 0., 0., 0.0617779, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.015487, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,      0.,
                                     0., 0., 0., 0.,        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.203364},
            .expectedEulerSource = {0., -1966.14, 0},
            .expectedDensityYiSource = {2.62257e-07,  3.01397e-09, 4.09717e-08,  -0.000291841, 1.13109e-07, 1.32473e-05, 0.000272949, 7.49846e-06, 4.11287e-29, 5.21486e-19, 1.30009e-13,
                                        1.77129e-14,  0.000141599, -0.000159441, 6.28935e-09,  9.99219e-11, 6.50777e-12, 1.54219e-05, 7.74522e-15, 2.01303e-08, 5.9374e-08,  -7.11753e-27,
                                        7.43587e-16,  1.47109e-18, 1.55456e-09,  9.85081e-11,  5.90665e-08, 7.39121e-25, 3.67216e-16, 5.56643e-25, 3.60909e-19, 3.47387e-21, -4.90663e-25,
                                        1.15566e-24,  6.18335e-14, 1.15305e-16,  1.41443e-20,  1.63021e-10, 6.24732e-21, 2.21197e-26, 1.35958e-19, 2.22306e-23, 7.79272e-21, -2.94329e-25,
                                        -1.15818e-24, 1.82192e-24, 4.16812e-24,  -2.70267e-24, 1.40856e-18, 2.57528e-14, 1.14893e-17, 1.55882e-16, -1.0375e-10}},
        (TCComputeSourceTestParameters){
            .mechFile = "inputs/eos/gri30.yaml",
            .dt = 0.017418748136926492,
            .inputEulerValues = {0.280629, 214342., 0.},
            .inputDensityYiValues = {2.70155e-06, 2.42588e-10, 1.75298e-09, 0.0615735,    5.91967e-09, 0.00013291,  1.42223e-06, 2.69273e-07, 1.17659e-25, 2.62694e-19, 1.04261e-12,
                                     1.55473e-13, 3.29875e-06, 0.0153352,   3.5785e-05,   2.61125e-07, 2.32785e-10, 0.000118819, 2.02248e-12, 3.19032e-09, 1.6112e-06,  3.70467e-18,
                                     1.90909e-09, 1.00394e-12, 3.84067e-06, 1.46041e-09,  5.52161e-05, 1.51027e-14, 3.77118e-08, 8.45969e-14, 1.76002e-20, 3.66826e-19, 2.92689e-20,
                                     3.18488e-20, 4.77626e-15, 1.73259e-15, 1.22235e-15,  1.81966e-10, 7.66494e-19, 1.00758e-26, 1.13374e-17, 2.26247e-22, 3.89214e-21, 2.08805e-21,
                                     1.82355e-22, 2.25953e-19, 1.26537e-19, -4.31761e-27, 6.78129e-13, 1.13467e-08, 8.23985e-12, 1.12011e-10, 0.203364},
            .expectedEulerSource = {0., 710973., 0.},
            .expectedDensityYiSource = {0.00155576,  2.04165e-07, 9.48011e-07, -0.0772463,   3.76528e-06, 0.0526686,   0.000375605, 0.000127858, 5.52447e-20, 8.30561e-15, 2.15501e-09,
                                        3.34547e-10, 0.000374787, -0.0504404,  0.0295509,    0.000613861, 3.23639e-07, 0.0229248,   5.17743e-09, 2.5618e-06,  0.000749697, 2.18505e-13,
                                        7.71297e-06, 8.76474e-09, 0.0042147,   3.01428e-06,  0.0143082,   8.95778e-10, 0.000171936, 2.11932e-09, 3.98707e-17, 2.93971e-15, 3.98326e-16,
                                        2.19774e-15, 4.30824e-12, 3.73653e-12, 4.39779e-12,  5.44988e-08, 5.9704e-15,  1.24544e-21, 9.80878e-14, 1.50217e-18, 1.15244e-16, 7.52342e-17,
                                        7.10193e-18, 4.27329e-15, 2.96867e-16, -6.83549e-25, 2.57715e-09, 3.0537e-05,  5.93473e-08, 9.20704e-07, -3.46948e-08}}));
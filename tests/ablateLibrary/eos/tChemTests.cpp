#include <numeric>
#include "PetscTestFixture.hpp"
#include "eos/tChem.hpp"
#include "gtest/gtest.h"

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS create and view tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemCreateAndViewParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::string expectedView;
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
    ASSERT_EQ(outputString, GetParam().expectedView);
}

INSTANTIATE_TEST_SUITE_P(
    TChemTests, TChemCreateAndViewFixture,
    testing::Values((TChemCreateAndViewParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                   .thermoFile = "inputs/eos/thermo30.dat",
                                                   .expectedView = "EOS: TChem\n\tmechFile: \"inputs/eos/grimech30.dat\"\n\tthermoFile: \"inputs/eos/thermo30.dat\"\n\tnumberSpecies: 53\n"},
                    (TChemCreateAndViewParameters){.mechFile = "inputs/eos/gri30.yaml", .expectedView = "EOS: TChem\n\tmechFile: \"inputs/eos/gri30.yaml\"\n\tnumberSpecies: 53\n"}),
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
    auto species = eos->GetSpecies();

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
                            eos->GetSpecies(),
                            params.yiMap,
                            params.conservedEulerValues[0],
                            conservedValues);

    // compute the reference temperature for other calculations
    auto temperatureFunction = eos->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, params.fields);
    PetscReal computedTemperature;
    PetscErrorCode ierr = temperatureFunction.function(conservedValues.data(), &computedTemperature, temperatureFunction.context.get());
    ASSERT_EQ(ierr, 0);

    if (params.expectedTemperature) {
        ASSERT_LT(PetscAbs(computedTemperature - params.expectedTemperature.value()) / params.expectedTemperature.value(), 1E-5)
            << "The percent difference for computed temperature (" << params.expectedTemperature.value() << " vs " << computedTemperature << ") should be small";
    }

    // Check each of the provided property
    for (const auto& [thermodynamicProperty, expectedValue] : params.testProperties) {
        // act/assert check for compute without temperature
        auto thermodynamicFunction = eos->GetThermodynamicFunction(thermodynamicProperty, params.fields);
        std::vector<PetscReal> computedProperty(expectedValue.size(), NAN);
        ierr = thermodynamicFunction.function(conservedValues.data(), computedProperty.data(), thermodynamicFunction.context.get());
        ASSERT_EQ(ierr, 0);
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
        ierr = thermodynamicTemperatureFunction.function(conservedValues.data(), computedTemperature, computedProperty.data(), thermodynamicTemperatureFunction.context.get());

        ASSERT_EQ(ierr, 0);
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
    auto yi = GetMassFraction(eos->GetSpecies(), GetParam().yiMap);

    // get the test params
    const auto& params = GetParam();
    std::vector<PetscReal> actualEulerValue(params.expectedEulerValue.size(), NAN);
    std::vector<PetscReal> actualDensityYiValue(eos->GetSpecies().size(), NAN);

    // act
    auto stateEulerFunction = eos->GetFieldFunctionFunction("euler", params.property1, params.property2);
    stateEulerFunction(params.property1Value, params.property2Value, params.velocity.size(), params.velocity.data(), yi.data(), actualEulerValue.data());
    auto stateDensityYiFunction = eos->GetFieldFunctionFunction("densityYi", params.property1, params.property2);
    stateDensityYiFunction(params.property1Value, params.property2Value, params.velocity.size(), params.velocity.data(), yi.data(), actualDensityYiValue.data());

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

INSTANTIATE_TEST_SUITE_P(TChemV1Tests, TChemFieldFunctionTestFixture,
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

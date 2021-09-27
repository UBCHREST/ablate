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
                         testing::Values((StiffenedGasEOSTestCreateAndViewParameters){.options = {},
                                                                                      .expectedView = "EOS: stiffenedGas\n\tgamma: 2.4\n\tCv: 3030\n\tp0: 1e+07\n\tT0: 584.25\n\te0: 1.393e+06\n"},
                                         (StiffenedGasEOSTestCreateAndViewParameters){.options = {{"gamma", "3.2"}, {"Cv", "100.2"}, {"p0", "3.5e6"}, {"T0", "654.32"}, {"e0", "1482000.9"}},
                                                                                      .expectedView = "EOS: stiffenedGas\n\tgamma: 3.2\n\tCv: 100.2\n\tp0: 3.5e+06\n\tT0: 654.32\n\te0: 1.482e+06\n"},
                                         (StiffenedGasEOSTestCreateAndViewParameters){
                                             .options = {{"gamma", "3.2"}, {"Cv", "100.2"}, {"p0", "3.5e6"}, {"T0", "654.32"}, {"e0", "1482000.9"}},  // need to replace with real state for expected
                                                                                                                                                      // internal energy, speed of sound, pressure
                                             .species = {"O2", "N2"},
                                             .expectedView = "EOS: stiffenedGas\n\tgamma: 3.2\n\tCv: 100.2\n\tp0: 3.5e+06\n\tT0: 654.32\n\te0: 1.482e+06\n\tspecies: O2, N2\n"}),
                         [](const testing::TestParamInfo<StiffenedGasEOSTestCreateAndViewParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS decode state tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct StiffenedGasEOSTestDecodeStateParameters {
    std::map<std::string, std::string> options;
    std::vector<PetscReal> densityYiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> velocityIn;
    PetscReal expectedInternalEnergy;
    PetscReal expectedSpeedOfSound;
    PetscReal expectedPressure;
};

class StiffenedGasTestDecodeStateFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<StiffenedGasEOSTestDecodeStateParameters> {};

TEST_P(StiffenedGasTestDecodeStateFixture, ShouldDecodeState) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters);

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal internalEnergy;
    PetscReal speedOfSound;
    PetscReal pressure;

    // act
    PetscErrorCode ierr = eos->GetDecodeStateFunction()(
        params.velocityIn.size(), params.densityIn, params.totalEnergyIn, &params.velocityIn[0], &params.densityYiIn[0], &internalEnergy, &speedOfSound, &pressure, eos->GetDecodeStateContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(internalEnergy, params.expectedInternalEnergy, 1E-6);
    ASSERT_NEAR(speedOfSound, params.expectedSpeedOfSound, 1E-6);
    ASSERT_NEAR(pressure, params.expectedPressure, 1E-6);
}

INSTANTIATE_TEST_SUITE_P(StiffenedGasEOSTests, StiffenedGasTestDecodeStateFixture,
                         testing::Values((StiffenedGasEOSTestDecodeStateParameters){.options = {{"gamma", "2.4"}, {"Cv", "3030.0"}, {"p0", "1.0e7"}, {"T0", "584.25"}, {"e0", "1393000.0"}},
                                                                                    .densityYiIn = {},
                                                                                    .densityIn = 998.7,
                                                                                    .totalEnergyIn = 1E5,
                                                                                    .velocityIn = {10, -20, 30},
                                                                                    .expectedInternalEnergy = 99300,
                                                                                    .expectedSpeedOfSound = 547.7264492,
                                                                                    .expectedPressure = 114839274},
                                         (StiffenedGasEOSTestDecodeStateParameters){.options = {{"gamma", "3.2"}, {"Cv", "100.2"}, {"p0", "3.5e6"}, {"T0", "654.32"}, {"e0", "1482000.9"}},
                                                                                    .densityYiIn = {},
                                                                                    .densityIn = 800,
                                                                                    .totalEnergyIn = 1.2E5,
                                                                                    .velocityIn = {0.0},
                                                                                    .expectedInternalEnergy = 1.2E+05,
                                                                                    .expectedSpeedOfSound = 902.2194855,
                                                                                    .expectedPressure = 2e8}),
                         [](const testing::TestParamInfo<StiffenedGasEOSTestDecodeStateParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS get temperature tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct StiffenedGasEOSTestTemperatureParameters {
    std::map<std::string, std::string> options;
    std::vector<PetscReal> densityYiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> massFluxIn;
    PetscReal expectedTemperature;
};

class StiffenedGasTestTemperatureFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<StiffenedGasEOSTestTemperatureParameters> {};

TEST_P(StiffenedGasTestTemperatureFixture, ShouldComputeTemperature) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters);

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal temperature;

    // act
    PetscErrorCode ierr = eos->GetComputeTemperatureFunction()(
        params.massFluxIn.size(), params.densityIn, params.totalEnergyIn, &params.massFluxIn[0], &params.densityYiIn[0], &temperature, eos->GetComputeTemperatureContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(temperature, params.expectedTemperature, 1E-6);
}

INSTANTIATE_TEST_SUITE_P(StiffenedGasEOSTests, StiffenedGasTestTemperatureFixture,
                         testing::Values((StiffenedGasEOSTestTemperatureParameters){.options = {{"gamma", "2.4"}, {"Cv", "3030.0"}, {"p0", "1.0e7"}, {"T0", "584.25"}, {"e0", "1393000.0"}},
                                                                                    .densityYiIn = {},
                                                                                    .densityIn = 998.7,
                                                                                    .totalEnergyIn = 1.5E+05,
                                                                                    .massFluxIn = {998.7 * 10, -998.7 * 20, 998.7 * 30},
                                                                                    .expectedTemperature = 173.7879538},
                                         (StiffenedGasEOSTestTemperatureParameters){.options = {{"gamma", "3.2"}, {"Cv", "100.2"}, {"p0", "3.5e6"}, {"T0", "654.32"}, {"e0", "1482000.9"}},
                                                                                    .densityYiIn = {},
                                                                                    .densityIn = 800,
                                                                                    .totalEnergyIn = 1.45E6,
                                                                                    .massFluxIn = {0.0},
                                                                                    .expectedTemperature = 334.9497405}),
                         [](const testing::TestParamInfo<StiffenedGasEOSTestTemperatureParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS get species tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(StiffenedGasEOSTests, StiffenedGasShouldReportNoSpeciesByDefault) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters);

    // act
    auto species = eos->GetSpecies();

    // assert
    ASSERT_EQ(0, species.size());
}

TEST(StiffenedGasEOSTests, StiffenedGasShouldReportSpeciesWhenProvided) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters, std::vector<std::string>{"N2", "H2"});

    // act
    auto species = eos->GetSpecies();

    // assert
    ASSERT_EQ(2, species.size());
    ASSERT_EQ("N2", species[0]);
    ASSERT_EQ("H2", species[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS get species enthalpy
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(StiffenedGasEOSTests, ShouldAssumeNoSpeciesEnthalpy) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters, std::vector<std::string>{"O2", "CH4", "N2"});

    std::vector<PetscReal> hiResult(3, 1);

    // act
    auto iErr = eos->GetComputeSpeciesSensibleEnthalpyFunction()(NAN, &hiResult[0], eos->GetComputeSpeciesSensibleEnthalpyContext());

    // assert
    ASSERT_EQ(0, iErr);
    auto expected = std::vector<PetscReal>{0.0, 0.0, 0.0};
    ASSERT_EQ(hiResult, expected);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Stiffened Gas DensityFunctionFromTemperaturePressure
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct StiffenedGasTestComputeDensityParameters {
    std::map<std::string, std::string> options;
    PetscReal temperatureIn;
    PetscReal pressureIn;
    PetscReal expectedDensity;
};

class StiffenedGasTestComputeDensityTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<StiffenedGasTestComputeDensityParameters> {};

TEST_P(StiffenedGasTestComputeDensityTestFixture, ShouldComputeCorrectTemperature) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters);

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

INSTANTIATE_TEST_SUITE_P(StiffenedGasEOSTests, StiffenedGasTestComputeDensityTestFixture,
                         testing::Values((StiffenedGasTestComputeDensityParameters){.options = {{"gamma", "2.4"}, {"Cv", "3030.0"}, {"p0", "1.0e7"}, {"T0", "584.25"}, {"e0", "1393000.0"}},
                                                                                    .temperatureIn = 300.0,
                                                                                    .pressureIn = 101325.0,
                                                                                    .expectedDensity = 32.37634695},
                                         (StiffenedGasTestComputeDensityParameters){.options = {{"gamma", "3.2"}, {"Cv", "100.2"}, {"p0", "3.5e6"}, {"T0", "654.32"}, {"e0", "1482000.9"}},
                                                                                    .temperatureIn = 1000.0,
                                                                                    .pressureIn = 1013250.0,
                                                                                    .expectedDensity = 3.660383784}),
                         [](const testing::TestParamInfo<StiffenedGasTestComputeDensityParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Stiffened Gas DensityFunctionFromTemperaturePressure
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct StiffenedGasComputeSensibleInternalEnergyParameters {
    std::map<std::string, std::string> options;
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal expectedSensibleInternalEnergy;
};

class StiffenedGasComputeSensibleInternalEnergyTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<StiffenedGasComputeSensibleInternalEnergyParameters> {};

TEST_P(StiffenedGasComputeSensibleInternalEnergyTestFixture, ShouldComputeCorrectEnergy) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters);

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal internalEnergy;

    // act
    PetscErrorCode ierr = eos->GetComputeSensibleInternalEnergyFunction()(params.temperatureIn, params.densityIn, nullptr, &internalEnergy, eos->GetComputeSensibleInternalEnergyContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(internalEnergy, params.expectedSensibleInternalEnergy, 1E-3);
}

INSTANTIATE_TEST_SUITE_P(StiffenedGasEOSTests, StiffenedGasComputeSensibleInternalEnergyTestFixture,
                         testing::Values((StiffenedGasComputeSensibleInternalEnergyParameters){.options = {{"gamma", "3.2"}, {"Cv", "100.2"}, {"p0", "3.5e6"}, {"T0", "654.32"}, {"e0", "1482000.9"}},
                                                                                               .temperatureIn = 39000,
                                                                                               .densityIn = 800,
                                                                                               .expectedSensibleInternalEnergy = 5324238.036},
                                         (StiffenedGasComputeSensibleInternalEnergyParameters){.options = {{"gamma", "2.4"}, {"Cv", "3030.0"}, {"p0", "1.0e7"}, {"T0", "584.25"}, {"e0", "1393000.0"}},
                                                                                               .temperatureIn = 350.0,
                                                                                               .densityIn = 998.7,
                                                                                               .expectedSensibleInternalEnergy = 683222.50},
                                         (StiffenedGasComputeSensibleInternalEnergyParameters){.options = {{"gamma", "2.4"}, {"Cv", "3030.0"}, {"p0", "1.0e7"}, {"T0", "584.25"}, {"e0", "1393000.0"}},
                                                                                               .temperatureIn = 350.0,
                                                                                               .densityIn = 20.1,
                                                                                               .expectedSensibleInternalEnergy = 683222.50}),
                         [](const testing::TestParamInfo<StiffenedGasComputeSensibleInternalEnergyParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Stiffened Gas ComputeSpecificHeatConstantPressure
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct StiffenedGasComputeSpecificHeatConstantPressureParameters {
    std::map<std::string, std::string> options;
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal expectedCp;
};

class StiffenedGasComputeSpecificHeatConstantPressureTestFixture : public testingResources::PetscTestFixture,
                                                                   public ::testing::WithParamInterface<StiffenedGasComputeSpecificHeatConstantPressureParameters> {};

TEST_P(StiffenedGasComputeSpecificHeatConstantPressureTestFixture, ShouldComputeCorrectEnergy) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::StiffenedGas>(parameters);

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal cp;

    // act
    PetscErrorCode ierr = eos->GetComputeSpecificHeatConstantPressureFunction()(params.temperatureIn, params.densityIn, nullptr, &cp, eos->GetComputeSensibleInternalEnergyContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(cp, params.expectedCp, 1E-3);
}

INSTANTIATE_TEST_SUITE_P(
    StiffenedGasEOSTests, StiffenedGasComputeSpecificHeatConstantPressureTestFixture,
    testing::Values((StiffenedGasComputeSpecificHeatConstantPressureParameters){.options = {{"gamma", "3.2"}, {"Cv", "100.2"}, {"p0", "3.5e6"}, {"T0", "654.32"}, {"e0", "1482000.9"}},
                                                                                .temperatureIn = NAN,
                                                                                .densityIn = NAN,
                                                                                .expectedCp = 320.64},
                    (StiffenedGasComputeSpecificHeatConstantPressureParameters){
                        .options = {{"gamma", "2.4"}, {"Cv", "3030.0"}, {"p0", "1.0e7"}, {"T0", "584.25"}, {"e0", "1393000.0"}}, .temperatureIn = NAN, .densityIn = NAN, .expectedCp = 7272.0}),
    [](const testing::TestParamInfo<StiffenedGasComputeSpecificHeatConstantPressureParameters>& info) { return std::to_string(info.index); });
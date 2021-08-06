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
/// EOS decode state tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestDecodeStateParameters {
    std::map<std::string, std::string> options;
    std::vector<PetscReal> densityYiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> velocityIn;
    PetscReal expectedInternalEnergy;
    PetscReal expectedSpeedOfSound;
    PetscReal expectedPressure;
};

class PerfectGasTestDecodeStateFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestDecodeStateParameters> {};

TEST_P(PerfectGasTestDecodeStateFixture, ShouldDecodeState) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters);

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

INSTANTIATE_TEST_SUITE_P(PerfectGasEOSTests, PerfectGasTestDecodeStateFixture,
                         testing::Values((EOSTestDecodeStateParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                                        .densityYiIn = {},
                                                                        .densityIn = 1.2,
                                                                        .totalEnergyIn = 1E5,
                                                                        .velocityIn = {10, -20, 30},
                                                                        .expectedInternalEnergy = 99300,
                                                                        .expectedSpeedOfSound = 235.8134856,
                                                                        .expectedPressure = 47664},
                                         (EOSTestDecodeStateParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                                        .densityYiIn = {},
                                                                        .densityIn = .9,
                                                                        .totalEnergyIn = 1.56E5,
                                                                        .velocityIn = {0.0},
                                                                        .expectedInternalEnergy = 1.56E+05,
                                                                        .expectedSpeedOfSound = 558.5696018,
                                                                        .expectedPressure = 140400}),
                         [](const testing::TestParamInfo<EOSTestDecodeStateParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS get temperature tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestTemperatureParameters {
    std::map<std::string, std::string> options;
    std::vector<PetscReal> densityYiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> massFluxIn;
    PetscReal expectedTemperature;
};

class PerfectGasTestTemperatureFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestTemperatureParameters> {};

TEST_P(PerfectGasTestTemperatureFixture, ShouldComputeTemperature) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters);

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

INSTANTIATE_TEST_SUITE_P(
    PerfectGasEOSTests, PerfectGasTestTemperatureFixture,
    testing::Values((EOSTestTemperatureParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                   .densityYiIn = {},
                                                   .densityIn = 1.2,
                                                   .totalEnergyIn = 1.50E+05,
                                                   .massFluxIn = {1.2 * 10, -1.2 * 20, 1.2 * 30},
                                                   .expectedTemperature = 208.0836237},
                    (EOSTestTemperatureParameters){
                        .options = {{"gamma", "2.0"}, {"Rgas", "4.0"}}, .densityYiIn = {}, .densityIn = .9, .totalEnergyIn = 1.56E5, .massFluxIn = {0.0}, .expectedTemperature = 39000}),
    [](const testing::TestParamInfo<EOSTestTemperatureParameters>& info) { return std::to_string(info.index); });

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
/// EOS get species enthalpy
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(PerfectGasEOSTests, ShouldAssumeNoSpeciesEnthalpy) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters, std::vector<std::string>{"O2", "CH4", "N2"});

    std::vector<PetscReal> hiResult(3, 1);

    // act
    auto iErr = eos->GetComputeSpeciesSensibleEnthalpyFunction()(NAN, &hiResult[0], eos->GetComputeSpeciesSensibleEnthalpyContext());

    // assert
    ASSERT_EQ(0, iErr);
    auto expected = std::vector<PetscReal>{0.0, 0.0, 0.0};
    ASSERT_EQ(hiResult, expected);
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

class PerfectGasTestComputeDensityTextFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<PerfectGasTestComputeDensityParameters> {};

TEST_P(PerfectGasTestComputeDensityTextFixture, ShouldComputeCorrectTemperature) {
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
    PerfectGasEOSTests, PerfectGasTestComputeDensityTextFixture,
    testing::Values((PerfectGasTestComputeDensityParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}}, .temperatureIn = 300.0, .pressureIn = 101325.0, .expectedDensity = 1.17682},
                    (PerfectGasTestComputeDensityParameters){.options = {{"gamma", "1.4"}, {"Rgas", "487.0"}}, .temperatureIn = 1000.0, .pressureIn = 1013250.0, .expectedDensity = 2.08059}),
    [](const testing::TestParamInfo<PerfectGasTestComputeDensityParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Perfect Gas DensityFunctionFromTemperaturePressure
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ComputeSensibleInternalEnergyParameters {
    std::map<std::string, std::string> options;
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal expectedSensibleInternalEnergy;
};

class ComputeSensibleInternalEnergyTextFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<ComputeSensibleInternalEnergyParameters> {};

TEST_P(ComputeSensibleInternalEnergyTextFixture, ShouldComputeCorrectEnergy) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters);

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

INSTANTIATE_TEST_SUITE_P(
    PerfectGasEOSTests, ComputeSensibleInternalEnergyTextFixture,
    testing::Values((ComputeSensibleInternalEnergyParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}}, .temperatureIn = 39000, .densityIn = .9, .expectedSensibleInternalEnergy = 1.56E5},
                    (ComputeSensibleInternalEnergyParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}}, .temperatureIn = 350.0, .densityIn = 1.1, .expectedSensibleInternalEnergy = 251125.00},
                    (ComputeSensibleInternalEnergyParameters){
                        .options = {{"gamma", "1.4"}, {"Rgas", "287.0"}}, .temperatureIn = 350.0, .densityIn = 20.1, .expectedSensibleInternalEnergy = 251125.00}),
    [](const testing::TestParamInfo<ComputeSensibleInternalEnergyParameters>& info) { return std::to_string(info.index); });
#include "PetscTestFixture.hpp"
#include "PetscTestViewer.hpp"
#include "eos.h"
#include "gtest/gtest.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS create and view tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestCreateAndViewParameters {
    std::string eosType;
    std::map<std::string, std::string> options;
    std::string expectedView;
};

class EOSTestCreateAndViewFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestCreateAndViewParameters> {};

TEST_P(EOSTestCreateAndViewFixture, ShouldCreateAndView) {
    // arrange
    EOSData eos;
    EOSCreate(&eos) >> errorChecker;

    // If provided, set the eosType
    if (!GetParam().eosType.empty()) {
        EOSSetType(eos, GetParam().eosType.c_str()) >> errorChecker;
    }

    // Define a set of options
    PetscOptions options;
    PetscOptionsCreate(&options) >> errorChecker;
    // add each option in the map
    for (const auto& option : GetParam().options) {
        PetscOptionsSetValue(options, option.first.c_str(), option.second.c_str()) >> errorChecker;
    }

    // setup the eos from options
    EOSSetOptions(eos, options) >> errorChecker;
    EOSSetFromOptions(eos) >> errorChecker;

    // setup a test viewer
    testingResources::PetscTestViewer viewer;

    // act
    EOSView(eos, viewer.GetViewer());

    // assert the output is as expected
    auto outputString = viewer.GetString();
    ASSERT_EQ(outputString, GetParam().expectedView);

    // cleanup
    EOSDestroy(&eos) >> errorChecker;
}

INSTANTIATE_TEST_SUITE_P(EOSTests, EOSTestCreateAndViewFixture,
                         testing::Values((EOSTestCreateAndViewParameters){.eosType = "perfectGas", .options = {}, .expectedView = "EOS: perfectGas\n  gamma: 1.400000\n  Rgas: 287.000000\n"},
                                         (EOSTestCreateAndViewParameters){
                                             .eosType = "perfectGas", .options = {{"-gamma", "3.2"}, {"-Rgas", "100.2"}}, .expectedView = "EOS: perfectGas\n  gamma: 3.200000\n  Rgas: 100.200000\n"}),
                         [](const testing::TestParamInfo<EOSTestCreateAndViewParameters>& info) { return std::to_string(info.index) + "_" + info.param.eosType; });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS decode state tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestDecodeStateParameters {
    std::string eosType;
    std::map<std::string, std::string> options;
    std::vector<PetscReal> yiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> velocityIn;
    PetscReal expectedInternalEnergy;
    PetscReal expectedSpeedOfSound;
    PetscReal expectedPressure;
};

class EOSTestDecodeStateFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestDecodeStateParameters> {};

TEST_P(EOSTestDecodeStateFixture, ShouldDecodeState) {
    // arrange
    EOSData eos;
    EOSCreate(&eos) >> errorChecker;

    // If provided, set the eosType
    if (!GetParam().eosType.empty()) {
        EOSSetType(eos, GetParam().eosType.c_str()) >> errorChecker;
    }

    // Define a set of options
    PetscOptions options;
    PetscOptionsCreate(&options) >> errorChecker;
    // add each option in the map
    for (const auto& option : GetParam().options) {
        PetscOptionsSetValue(options, option.first.c_str(), option.second.c_str()) >> errorChecker;
    }

    // setup the eos from options
    EOSSetOptions(eos, options) >> errorChecker;
    EOSSetFromOptions(eos) >> errorChecker;

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal internalEnergy;
    PetscReal speedOfSound;
    PetscReal pressure;

    // act
    // EOSDecodeState(EOSData eos, const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* p);
    PetscErrorCode ierr = EOSDecodeState(eos, &params.yiIn[0], params.velocityIn.size(), params.densityIn, params.totalEnergyIn, &params.velocityIn[0], &internalEnergy, &speedOfSound, &pressure);

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(internalEnergy, params.expectedInternalEnergy, 1E-6);
    ASSERT_NEAR(speedOfSound, params.expectedSpeedOfSound, 1E-6);
    ASSERT_NEAR(pressure, params.expectedPressure, 1E-6);

    // cleanup
    EOSDestroy(&eos) >> errorChecker;
}

INSTANTIATE_TEST_SUITE_P(EOSTests, EOSTestDecodeStateFixture,
                         testing::Values((EOSTestDecodeStateParameters){.eosType = "perfectGas",
                                                                        .options = {{"-gamma", "1.4"}, {"-Rgas", "287.0"}},
                                                                        .yiIn = {},
                                                                        .densityIn = 1.2,
                                                                        .totalEnergyIn = 1E5,
                                                                        .velocityIn = {10, -20, 30},
                                                                        .expectedInternalEnergy = 99300,
                                                                        .expectedSpeedOfSound = 235.8134856,
                                                                        .expectedPressure = 47664},
                                         (EOSTestDecodeStateParameters){.eosType = "perfectGas",
                                                                        .options = {{"-gamma", "2.0"}, {"-Rgas", "4.0"}},
                                                                        .yiIn = {},
                                                                        .densityIn = .9,
                                                                        .totalEnergyIn = 1.56E5,
                                                                        .velocityIn = {0.0},
                                                                        .expectedInternalEnergy = 1.56E+05,
                                                                        .expectedSpeedOfSound = 558.5696018,
                                                                        .expectedPressure = 140400}),
                         [](const testing::TestParamInfo<EOSTestDecodeStateParameters>& info) { return std::to_string(info.index) + "_" + info.param.eosType; });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS decode state tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestTemperatureParameters {
    std::string eosType;
    std::map<std::string, std::string> options;
    std::vector<PetscReal> yiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> massFluxIn;
    PetscReal expectedTemperature;
};

class EOSTestTemperatureFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestTemperatureParameters> {};

TEST_P(EOSTestTemperatureFixture, ShouldComputeTemperature) {
    // arrange
    EOSData eos;
    EOSCreate(&eos) >> errorChecker;

    // If provided, set the eosType
    if (!GetParam().eosType.empty()) {
        EOSSetType(eos, GetParam().eosType.c_str()) >> errorChecker;
    }

    // Define a set of options
    PetscOptions options;
    PetscOptionsCreate(&options) >> errorChecker;
    // add each option in the map
    for (const auto& option : GetParam().options) {
        PetscOptionsSetValue(options, option.first.c_str(), option.second.c_str()) >> errorChecker;
    }

    // setup the eos from options
    EOSSetOptions(eos, options) >> errorChecker;
    EOSSetFromOptions(eos) >> errorChecker;

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal temperature;

    // act
    PetscErrorCode ierr = EOSTemperature(eos, &params.yiIn[0], params.massFluxIn.size(), params.densityIn, params.totalEnergyIn, &params.massFluxIn[0], &temperature);

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(temperature, params.expectedTemperature, 1E-6);

    // cleanup
    EOSDestroy(&eos) >> errorChecker;
}

INSTANTIATE_TEST_SUITE_P(
    EOSTests, EOSTestTemperatureFixture,
    testing::Values(
        (EOSTestTemperatureParameters){.eosType = "perfectGas",
                                       .options = {{"-gamma", "1.4"}, {"-Rgas", "287.0"}},
                                       .yiIn = {},
                                       .densityIn = 1.2,
                                       .totalEnergyIn = 1.50E+05,
                                       .massFluxIn = {1.2 * 10, -1.2 * 20, 1.2 * 30},
                                       .expectedTemperature = 208.0836237},
        (EOSTestTemperatureParameters){
            .eosType = "perfectGas", .options = {{"-gamma", "2.0"}, {"-Rgas", "4.0"}}, .yiIn = {}, .densityIn = .9, .totalEnergyIn = 1.56E5, .massFluxIn = {0.0}, .expectedTemperature = 39000}),
    [](const testing::TestParamInfo<EOSTestTemperatureParameters>& info) { return std::to_string(info.index) + "_" + info.param.eosType; });

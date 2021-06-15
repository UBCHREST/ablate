#include "PetscTestFixture.hpp"
#include "eos/perfectGas.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS create and view tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestCreateAndViewParameters {
    std::map<std::string, std::string> options;
    std::string expectedView;
};

class PerfectGasTestCreateAndViewFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestCreateAndViewParameters> {};

TEST_P(PerfectGasTestCreateAndViewFixture, ShouldCreateAndView) {
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters);

    std::stringstream outputStream;

    // act
    outputStream << *eos;

    // assert the output is as expected
    auto outputString = outputStream.str();
    ASSERT_EQ(outputString, GetParam().expectedView);
}

INSTANTIATE_TEST_SUITE_P(EOSTests, PerfectGasTestCreateAndViewFixture,
                         testing::Values((EOSTestCreateAndViewParameters){.options = {}, .expectedView = "EOS: perfectGas\n\tgamma: 1.4\n\tRgas: 287\n"},
                                         (EOSTestCreateAndViewParameters){.options = {{"gamma", "3.2"}, {"Rgas", "100.2"}}, .expectedView = "EOS: perfectGas\n\tgamma: 3.2\n\tRgas: 100.2\n"}),
                         [](const testing::TestParamInfo<EOSTestCreateAndViewParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS decode state tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestDecodeStateParameters {
    std::map<std::string, std::string> options;
    std::vector<PetscReal> yiIn;
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
        &params.yiIn[0], params.velocityIn.size(), params.densityIn, params.totalEnergyIn, &params.velocityIn[0], &internalEnergy, &speedOfSound, &pressure, eos->GetDecodeStateContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(internalEnergy, params.expectedInternalEnergy, 1E-6);
    ASSERT_NEAR(speedOfSound, params.expectedSpeedOfSound, 1E-6);
    ASSERT_NEAR(pressure, params.expectedPressure, 1E-6);
}

INSTANTIATE_TEST_SUITE_P(EOSTests, PerfectGasTestDecodeStateFixture,
                         testing::Values((EOSTestDecodeStateParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                                        .yiIn = {},
                                                                        .densityIn = 1.2,
                                                                        .totalEnergyIn = 1E5,
                                                                        .velocityIn = {10, -20, 30},
                                                                        .expectedInternalEnergy = 99300,
                                                                        .expectedSpeedOfSound = 235.8134856,
                                                                        .expectedPressure = 47664},
                                         (EOSTestDecodeStateParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                                                                        .yiIn = {},
                                                                        .densityIn = .9,
                                                                        .totalEnergyIn = 1.56E5,
                                                                        .velocityIn = {0.0},
                                                                        .expectedInternalEnergy = 1.56E+05,
                                                                        .expectedSpeedOfSound = 558.5696018,
                                                                        .expectedPressure = 140400}),
                         [](const testing::TestParamInfo<EOSTestDecodeStateParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS decode state tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct EOSTestTemperatureParameters {
    std::map<std::string, std::string> options;
    std::vector<PetscReal> yiIn;
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
        &params.yiIn[0], params.massFluxIn.size(), params.densityIn, params.totalEnergyIn, &params.massFluxIn[0], &temperature, eos->GetComputeTemperatureContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(temperature, params.expectedTemperature, 1E-6);
}

INSTANTIATE_TEST_SUITE_P(EOSTests, PerfectGasTestTemperatureFixture,
                         testing::Values((EOSTestTemperatureParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                                                                        .yiIn = {},
                                                                        .densityIn = 1.2,
                                                                        .totalEnergyIn = 1.50E+05,
                                                                        .massFluxIn = {1.2 * 10, -1.2 * 20, 1.2 * 30},
                                                                        .expectedTemperature = 208.0836237},
                                         (EOSTestTemperatureParameters){
                                             .options = {{"gamma", "2.0"}, {"Rgas", "4.0"}}, .yiIn = {}, .densityIn = .9, .totalEnergyIn = 1.56E5, .massFluxIn = {0.0}, .expectedTemperature = 39000}),
                         [](const testing::TestParamInfo<EOSTestTemperatureParameters>& info) { return std::to_string(info.index); });

TEST(EOSTests, PerfectGasShouldReportNoSpecies){
    // arrange
    auto parameters = std::make_shared<ablate::parameters::MapParameters>();
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters);

    // act
    auto species = eos->GetSpecies();

    // assert
    ASSERT_EQ(0, species.size());
}
#include "PetscTestFixture.hpp"
#include "eos/tChem.hpp"
#include "gtest/gtest.h"

/*
 * Helper function to fill mass fraction
 */
static std::vector<PetscReal> GetDensityMassFraction(const std::vector<std::string>& species, const std::map<std::string, PetscReal>& yiIn, double density) {
    std::vector<PetscReal> yi(species.size(), 0.0);

    for (const auto& value : yiIn) {
        // Get the index
        auto it = std::find(species.begin(), species.end(), value.first);
        if (it != species.end()) {
            auto index = std::distance(species.begin(), it);

            yi[index] = value.second * density;
        }
    }
    return yi;
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

INSTANTIATE_TEST_SUITE_P(EOSTests, TChemCreateAndViewFixture,
                         testing::Values((TChemCreateAndViewParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                        .thermoFile = "inputs/eos/thermo30.dat",
                                                                        .expectedView = "EOS: TChemV1\n\tmechFile: \"inputs/eos/grimech30.dat\"\n\tthermoFile: \"inputs/eos/thermo30.dat\"\n"}),
                         [](const testing::TestParamInfo<TChemCreateAndViewParameters>& info) { return info.param.mechFile.stem().string() + "_" + info.param.thermoFile.stem().string(); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS Get Species Tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

INSTANTIATE_TEST_SUITE_P(EOSTests, TChemGetSpeciesFixture,
                         testing::Values((TChemGetSpeciesParameters){
                             .mechFile = "inputs/eos/grimech30.dat",
                             .thermoFile = "inputs/eos/thermo30.dat",
                             .expectedSpecies = {"H2",    "H",    "O",     "O2",  "OH",   "H2O",  "HO2",  "H2O2", "C",    "CH",   "CH2",   "CH2(S)", "CH3", "CH4",  "CO",   "CO2",    "HCO",   "CH2O",
                                                 "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2", "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH",  "N",   "NH",   "NH2",  "NH3",    "NNH",   "NO",
                                                 "NO2",   "N2O",  "HNO",   "CN",  "HCN",  "H2CN", "HCNN", "HCNO", "HOCN", "HNCO", "NCO",   "N2",     "AR",  "C3H7", "C3H8", "CH2CHO", "CH3CHO"}}),
                         [](const testing::TestParamInfo<TChemGetSpeciesParameters>& info) { return info.param.mechFile.stem().string() + "_" + info.param.thermoFile.stem().string(); });

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// EOS decode state tests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemStateParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::map<std::string, PetscReal> yiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> massFluxIn;
    PetscReal expectedTemperature;
    PetscReal expectedInternalEnergy;
    PetscReal expectedSpeedOfSound;
    PetscReal expectedPressure;
};

class TChemStateTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemStateParameters> {};

TEST_P(TChemStateTestFixture, ShouldDecodeState) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal internalEnergy;
    PetscReal speedOfSound;
    PetscReal pressure;

    // get the mass fraction as an array
    auto densityYi = GetDensityMassFraction(eos->GetSpecies(), params.yiIn, params.densityIn);

    // convert the massFrac in to velocity
    std::vector<double> velocityIn;
    for (const auto& rhoV : params.massFluxIn) {
        velocityIn.push_back(rhoV / params.densityIn);
    }

    // act
    PetscErrorCode ierr = eos->GetDecodeStateFunction()(
        velocityIn.size(), params.densityIn, params.totalEnergyIn, &velocityIn[0], &densityYi[0], &internalEnergy, &speedOfSound, &pressure, eos->GetDecodeStateContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(internalEnergy, params.expectedInternalEnergy, .1);
    ASSERT_NEAR(speedOfSound, params.expectedSpeedOfSound, .1);
    ASSERT_LT(PetscAbs(pressure - params.expectedPressure) / params.expectedPressure, 1E-5) << "The percent difference in pressure should be less than 1E-5";
}

TEST_P(TChemStateTestFixture, ShouldComputeTemperature) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // get the mass fraction as an array
    auto densityYi = GetDensityMassFraction(eos->GetSpecies(), params.yiIn, params.densityIn);

    // Prepare outputs
    PetscReal temperature;

    // act
    PetscErrorCode ierr =
        eos->GetComputeTemperatureFunction()(params.massFluxIn.size(), params.densityIn, params.totalEnergyIn, &params.massFluxIn[0], &densityYi[0], &temperature, eos->GetComputeTemperatureContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(temperature, params.expectedTemperature, 1E-2);
}

INSTANTIATE_TEST_SUITE_P(EOSTests, TChemStateTestFixture,
                         testing::Values((TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yiIn = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                .densityIn = 1.2,
                                                                .totalEnergyIn = 1.E+05,
                                                                .massFluxIn = {1.2 * 10, -1.2 * 20, 1.2 * 30},
                                                                .expectedTemperature = 499.25,
                                                                .expectedInternalEnergy = 99300.0,
                                                                .expectedSpeedOfSound = 464.33,
                                                                .expectedPressure = 197710.5},
                                         (TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yiIn = {{"O2", .3}, {"N2", .4}, {"CH2", .1}, {"NO", .2}},
                                                                .densityIn = 0.8,
                                                                .totalEnergyIn = 3.2E5,
                                                                .massFluxIn = {0, 0, 0},
                                                                .expectedTemperature = 762.664,
                                                                .expectedInternalEnergy = 320000.0,
                                                                .expectedSpeedOfSound = 560.83,
                                                                .expectedPressure = 189973.54},
                                         (TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yiIn = {{"N2", 1.0}},
                                                                .densityIn = 3.3,
                                                                .totalEnergyIn = 1000,
                                                                .massFluxIn = {0.0, 3.3 * 2, 3.3 * 4},
                                                                .expectedTemperature = 418.079,
                                                                .expectedInternalEnergy = 990.0,
                                                                .expectedSpeedOfSound = 416.04,
                                                                .expectedPressure = 409488.10},
                                         (TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yiIn = {{"H2", .35}, {"H2O", .35}, {"N2", .3}},
                                                                .densityIn = 0.01,
                                                                .totalEnergyIn = 1E5,
                                                                .massFluxIn = {.01 * -1, .01 * -2, .01 * -3},
                                                                .expectedTemperature = 437.46,
                                                                .expectedInternalEnergy = 99993.0,
                                                                .expectedSpeedOfSound = 1013.73,
                                                                .expectedPressure = 7411.11},
                                         (TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yiIn = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                .densityIn = 999.9,
                                                                .totalEnergyIn = 1E4,
                                                                .massFluxIn = {999.9 * -10, 999.9 * -20, 999.9 * -300},
                                                                .expectedTemperature = 394.59,
                                                                .expectedInternalEnergy = -35250.0,
                                                                .expectedSpeedOfSound = 623.9,
                                                                .expectedPressure = 281125963.5}),
                         [](const testing::TestParamInfo<TChemStateParameters>& info) { return std::to_string(info.index); });

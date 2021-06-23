#include "PetscTestFixture.hpp"
#include "eos/tChem.hpp"
#include "gtest/gtest.h"


/*
 * Helper function to fill mass fraction
 */
static std::vector<PetscReal> GetMassFraction(const std::vector<std::string>& species, const std::map<std::string, PetscReal>& yiIn){
     std::vector<PetscReal> yi(species.size(), 0.0);

     for(const auto& value: yiIn){
         // Get the index
         auto it = std::find(species.begin(), species.end(), value.first);
         if(it != species.end()){
             auto index = std::distance(species.begin(), it);

             yi[index] = value.second;
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
/// EOS create and view tests
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
struct TChemDecodeStateParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::vector<PetscReal> densityYiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> velocityIn;
    PetscReal expectedInternalEnergy;
    PetscReal expectedSpeedOfSound;
    PetscReal expectedPressure;
};

class TChemTestDecodeStateFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemDecodeStateParameters> {};

TEST_P(TChemTestDecodeStateFixture, ShouldDecodeState) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal internalEnergy;
    PetscReal speedOfSound;
    PetscReal pressure;

    // act
    PetscErrorCode ierr = eos->GetDecodeStateFunction()(
        &params.densityYiIn[0], params.velocityIn.size(), params.densityIn, params.totalEnergyIn, &params.velocityIn[0], &internalEnergy, &speedOfSound, &pressure, eos->GetDecodeStateContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(internalEnergy, params.expectedInternalEnergy, 1E-6);
    ASSERT_NEAR(speedOfSound, params.expectedSpeedOfSound, 1E-6);
    ASSERT_NEAR(pressure, params.expectedPressure, 1E-6);
}

INSTANTIATE_TEST_SUITE_P(EOSTests, TChemTestDecodeStateFixture,
                         testing::Values((TChemDecodeStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                      .thermoFile = "inputs/eos/thermo30.dat",
                                                                      .densityYiIn = {23., 323},
                                                                      .densityIn = 1.2,
                                                                      .totalEnergyIn = 1E5,
                                                                      .velocityIn = {10, -20, 30},
                                                                      .expectedInternalEnergy = 99300,
                                                                      .expectedSpeedOfSound = 464.3326095106185,
                                                                      .expectedPressure = 197709.15581272854}),
                         [](const testing::TestParamInfo<TChemDecodeStateParameters>& info) { return std::to_string(info.index); });

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// EOS decode state tests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemTemperatureParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::map<std::string, PetscReal> yiIn;
    PetscReal densityIn;
    PetscReal totalEnergyIn;
    std::vector<PetscReal> massFluxIn;
    PetscReal expectedTemperature;
};

class TChemTestTemperatureFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemTemperatureParameters> {};

TEST_P(TChemTestTemperatureFixture, ShouldComputeTemperature) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // get the mass fraction as an array
    auto yi = GetMassFraction(eos->GetSpecies(), params.yiIn);

    // Prepare outputs
    PetscReal temperature;

    // act
    PetscErrorCode ierr = eos->GetComputeTemperatureFunction()(
        &yi[0], params.massFluxIn.size(), params.densityIn, params.totalEnergyIn, &params.massFluxIn[0], &temperature, eos->GetComputeTemperatureContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(temperature, params.expectedTemperature, 1E-2);
}

INSTANTIATE_TEST_SUITE_P(EOSTests, TChemTestTemperatureFixture,
                         testing::Values((TChemTemperatureParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                      .thermoFile = "inputs/eos/thermo30.dat",
                                                                      .yiIn = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                      .densityIn = 1.2,
                                                                      .totalEnergyIn = 1.E+05,
                                                                      .massFluxIn = {1.2 * 10, -1.2 * 20, 1.2 * 30},
                                                                      .expectedTemperature = 499.25}),
                         [](const testing::TestParamInfo<TChemTemperatureParameters>& info) { return std::to_string(info.index); });

#include "PetscTestFixture.hpp"
#include "eos/tChem.hpp"
#include "gtest/gtest.h"

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
 struct EOSTestDecodeStateParameters {
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

 class TChemTestDecodeStateFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestDecodeStateParameters> {};

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
                          testing::Values((EOSTestDecodeStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                         .thermoFile = "inputs/eos/thermo30.dat",
                                                                         .densityYiIn = {},
                                                                         .densityIn = 1.2,
                                                                         .totalEnergyIn = 1E5,
                                                                         .velocityIn = {10, -20, 30},
                                                                         .expectedInternalEnergy = 99300,
                                                                         .expectedSpeedOfSound = 464.3326095106185,
                                                                         .expectedPressure = 197709.15581272854}),
                          [](const testing::TestParamInfo<EOSTestDecodeStateParameters>& info) { return std::to_string(info.index); });

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// EOS decode state tests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// struct EOSTestTemperatureParameters {
//     std::map<std::string, std::string> options;
//     std::vector<PetscReal> yiIn;
//     PetscReal densityIn;
//     PetscReal totalEnergyIn;
//     std::vector<PetscReal> massFluxIn;
//     PetscReal expectedTemperature;
// };
//
// class PerfectGasTestTemperatureFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<EOSTestTemperatureParameters> {};
//
// TEST_P(PerfectGasTestTemperatureFixture, ShouldComputeTemperature) {
//     // arrange
//     auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
//     std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::PerfectGas>(parameters);
//
//     // get the test params
//     const auto& params = GetParam();
//
//     // Prepare outputs
//     PetscReal temperature;
//
//     // act
//     PetscErrorCode ierr = eos->GetComputeTemperatureFunction()(
//         &params.yiIn[0], params.massFluxIn.size(), params.densityIn, params.totalEnergyIn, &params.massFluxIn[0], &temperature, eos->GetComputeTemperatureContext());
//
//     // assert
//     ASSERT_EQ(ierr, 0);
//     ASSERT_NEAR(temperature, params.expectedTemperature, 1E-6);
// }
//
// INSTANTIATE_TEST_SUITE_P(EOSTests, PerfectGasTestTemperatureFixture,
//                          testing::Values((EOSTestTemperatureParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
//                                                                         .yiIn = {},
//                                                                         .densityIn = 1.2,
//                                                                         .totalEnergyIn = 1.50E+05,
//                                                                         .massFluxIn = {1.2 * 10, -1.2 * 20, 1.2 * 30},
//                                                                         .expectedTemperature = 208.0836237},
//                                          (EOSTestTemperatureParameters){
//                                              .options = {{"gamma", "2.0"}, {"Rgas", "4.0"}}, .yiIn = {}, .densityIn = .9, .totalEnergyIn = 1.56E5, .massFluxIn = {0.0}, .expectedTemperature =
//                                              39000}),
//                          [](const testing::TestParamInfo<EOSTestTemperatureParameters>& info) { return std::to_string(info.index); });

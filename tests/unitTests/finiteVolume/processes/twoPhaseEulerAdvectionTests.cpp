#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <cmath>
#include <memory>
#include <vector>
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

using namespace ablate;

struct TwoPhaseEulerAdvectionTestDecodeStateParameters {
    std::string testName;
    std::shared_ptr<eos::EOS> eosGas;
    std::shared_ptr<eos::EOS> eosLiquid;
    PetscInt dim;
    std::vector<PetscReal> conservedValuesIn;
    std::vector<PetscReal> normalIn;
    PetscReal expectedDensity;
    PetscReal expectedDensityG;
    PetscReal expectedDensityL;
    PetscReal expectedNormalVelocity;
    std::vector<PetscReal> expectedVelocity;
    PetscReal expectedInternalEnergy;
    PetscReal expectedInternalEnergyG;
    PetscReal expectedInternalEnergyL;
    PetscReal expectedSoundSpeedG;
    PetscReal expectedSoundSpeedL;
    PetscReal expectedMG;
    PetscReal expectedML;
    PetscReal expectedPressure;
    PetscReal expectedAlpha;
};

class TwoPhaseEulerAdvectionTestDecodeStateFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TwoPhaseEulerAdvectionTestDecodeStateParameters> {};

TEST_P(TwoPhaseEulerAdvectionTestDecodeStateFixture, ShouldDecodeState) {
    // arrange
    auto eosGas = GetParam().eosGas;
    auto eosLiquid = GetParam().eosLiquid;

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal density;
    PetscReal densityG;
    PetscReal densityL;
    PetscReal normalVelocity;
    std::vector<PetscReal> velocity(3);
    PetscReal internalEnergy;
    PetscReal internalEnergyG;
    PetscReal internalEnergyL;
    PetscReal soundSpeedG;
    PetscReal soundSpeedL;
    PetscReal MG;
    PetscReal ML;
    PetscReal pressure;
    PetscReal temperature;
    PetscReal alpha;

    // compute offsets
    PetscInt uOff[2] = {2 + params.dim /*densityYi*/, 0 /*euler*/};

    // Create a two phase decoder
    auto decoder = finiteVolume::processes::TwoPhaseEulerAdvection::CreateTwoPhaseDecoder(params.dim, eosGas, eosLiquid);

    // act
    decoder->DecodeTwoPhaseEulerState(params.dim,
                                      uOff,
                                      params.conservedValuesIn.data(),
                                      &params.normalIn[0],
                                      &density,
                                      &densityG,
                                      &densityL,
                                      &normalVelocity,
                                      &velocity[0],
                                      &internalEnergy,
                                      &internalEnergyG,
                                      &internalEnergyL,
                                      &soundSpeedG,
                                      &soundSpeedL,
                                      &MG,
                                      &ML,
                                      &pressure,
                                      &temperature,
                                      &alpha);

    // assert
    ASSERT_NEAR(density, params.expectedDensity, 1E-6);
    ASSERT_NEAR(densityG, params.expectedDensityG, 1E-6);
    ASSERT_NEAR(densityL, params.expectedDensityL, 1E-6);
    ASSERT_NEAR(normalVelocity, params.expectedNormalVelocity, 1E-6);
    ASSERT_NEAR(velocity[0], params.expectedVelocity[0], 1E-6);
    ASSERT_NEAR(velocity[1], params.expectedVelocity[1], 1E-6);
    ASSERT_NEAR(velocity[2], params.expectedVelocity[2], 1E-6);
    ASSERT_NEAR(internalEnergy, params.expectedInternalEnergy, 1E-6);
    ASSERT_NEAR(internalEnergyG, params.expectedInternalEnergyG, params.expectedInternalEnergyG * 1E-6);
    ASSERT_NEAR(internalEnergyL, params.expectedInternalEnergyL, params.expectedInternalEnergyL * 1E-6);
    ASSERT_NEAR(soundSpeedG, params.expectedSoundSpeedG, 1E-6);
    ASSERT_NEAR(soundSpeedL, params.expectedSoundSpeedL, 1E-6);
    ASSERT_NEAR(MG, params.expectedMG, 1E-6);
    ASSERT_NEAR(ML, params.expectedML, 1E-6);
    ASSERT_NEAR(pressure, params.expectedPressure, 1E-2);
    ASSERT_NEAR(alpha, params.expectedAlpha, 1E-6);
}

INSTANTIATE_TEST_SUITE_P(
    TwoPhaseEulerAdvectionTests, TwoPhaseEulerAdvectionTestDecodeStateFixture,
    testing::Values(
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){
            .testName = "all_phase_1_near_boundary",
            .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
            .dim = 3,
            .conservedValuesIn = {3.9, 936986.7, 39.0, -78.0, 117.0, 3.9 - 1E-4},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
            .normalIn = {0.5, 0.5, 0.7071},                                        // x, y, z
            .expectedDensity = 3.9,
            .expectedDensityG = 3.899934912891986,
            .expectedDensityL = 11.170472255489022,
            .expectedNormalVelocity = 16.213,
            .expectedVelocity = {10.0, -20.0, 30.0},
            .expectedInternalEnergy = 239553.0,
            .expectedInternalEnergyG = 239558.75261655406,
            .expectedInternalEnergyL = 15206.707008032125,
            .expectedSoundSpeedG = 366.2688922980906,
            .expectedSoundSpeedL = 327.19293595147525,
            .expectedMG = 0.044265293452234906,
            .expectedML = 0.04955180328955663,
            .expectedPressure = 373705.4172072614,
            .expectedAlpha = 0.999991047827011},
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){
            .testName = "all_phase_2_near_boundary",
            .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
            .dim = 3,
            .conservedValuesIn = {3.9, 936986.7, 39.0, -78.0, 117.0, 0.0001},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
            .normalIn = {0.5, 0.5, 0.7071},                                    // x, y, z
            .expectedDensity = 3.9,
            .expectedDensityG = 1.3616678745644597,
            .expectedDensityL = 3.9001864271457074,
            .expectedNormalVelocity = 16.213,
            .expectedVelocity = {10.0, -20.0, 30.0},
            .expectedInternalEnergy = 239553.0,
            .expectedInternalEnergyG = 3772369.444636368,
            .expectedInternalEnergyL = 239462.41264020524,
            .expectedSoundSpeedG = 1453.4534354413854,
            .expectedSoundSpeedL = 1298.3895351500048,
            .expectedMG = 0.011154812121708206,
            .expectedML = 0.012487007605253758,
            .expectedPressure = 2054685.7134999656,
            .expectedAlpha = 7.343934733863484e-05},
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){
            .testName = "perfect_gas_plus_perfect_gas",
            .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
            .dim = 3,
            .conservedValuesIn = {3.9, 936986.7, 39.0, -78.0, 117.0, 0.8},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
            .normalIn = {0.5, 0.5, 0.7071},                                 // x, y, z
            .expectedDensity = 3.9,
            .expectedDensityG = 1.88229965,
            .expectedDensityL = 5.391417167,
            .expectedNormalVelocity = 16.213,
            .expectedVelocity = {10.0, -20.0, 30.0},
            .expectedInternalEnergy = 239553.0,
            .expectedInternalEnergyG = 937273.0745446227,
            .expectedInternalEnergyL = 59496.20656912966,
            .expectedSoundSpeedG = 724.4811396751392,
            .expectedSoundSpeedL = 647.1887624539481,
            .expectedMG = 0.02237877442505955,
            .expectedML = 0.0250514238512503,
            .expectedPressure = 705691.5126557435,
            .expectedAlpha = 0.425012032},
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){
            .testName = "perfect_gas_plus_stiffened_gas",
            .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .dim = 3,
            .conservedValuesIn = {340.4, 851000000.0, 3404.0, -6808.0, 10212.0, 0.8},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
            .normalIn = {0.5, 0.5, 0.7071},                                            // x, y, z
            .expectedDensity = 340.4,
            .expectedDensityG = 1.2352213315429055,
            .expectedDensityL = 963.8341087392608,
            .expectedNormalVelocity = 16.213,
            .expectedVelocity = {10.0, -20.0, 30.0},
            .expectedInternalEnergy = 2499300.0,
            .expectedInternalEnergyG = 222008.74346659306,
            .expectedInternalEnergyL = 2504664.64371386,
            .expectedSoundSpeedG = 352.5973572522801,
            .expectedSoundSpeedL = 1527.8918538107569,
            .expectedMG = 0.04598162653953118,
            .expectedML = 0.010611353126572876,
            .expectedPressure = 109691.97428758894,
            .expectedAlpha = 0.6476572089317193},
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){
            .testName = "perfect_gas_plus_stiffened_gas_all_water",
            .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .dim = 3,
            .conservedValuesIn = {994.0897497618486, 2414070815.4506435, 0.0, 0.0, 0.0, 0.0},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
            .normalIn = {0.5, 0.5, 0.7071},                                                    // x, y, z
            .expectedDensity = 994.0897497618486,
            .expectedDensityG = 1.1614401858216397,
            .expectedDensityL = 994.0897497618485,
            .expectedNormalVelocity = 0.0,
            .expectedVelocity = {0.0, 0.0, 0.0},
            .expectedInternalEnergy = 2428423.4054611027,
            .expectedInternalEnergyG = 215250.0000006027,
            .expectedInternalEnergyL = 2428423.4054611027,
            .expectedSoundSpeedG = 347.18870949432886,
            .expectedSoundSpeedL = 1504.4548407978218,
            .expectedMG = 0.0,
            .expectedML = 0.0,
            .expectedPressure = 99999.99999952316,
            .expectedAlpha = 0.0},
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){
            .testName = "perfect_gas_plus_stiffened_gas_all_air",
            .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .dim = 3,
            .conservedValuesIn = {1.1614401858304297, 250000.0, 0.0, 0.0, 0.0, 1.1614401858304297},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
            .normalIn = {0.5, 0.5, 0.7071},                                                          // x, y, z
            .expectedDensity = 1.1614401858304297,
            .expectedDensityG = 1.1614401858304295,
            .expectedDensityL = 994.089749761849,
            .expectedNormalVelocity = 0.0,
            .expectedVelocity = {0.0, 0.0, 0.0},
            .expectedInternalEnergy = 215250.0,
            .expectedInternalEnergyG = 215250.0,
            .expectedInternalEnergyL = 2428423.405461102,
            .expectedSoundSpeedG = 347.1887094938428,
            .expectedSoundSpeedL = 1504.4548407978218,
            .expectedMG = 0.0,
            .expectedML = 0.0,
            .expectedPressure = 100000.0,
            .expectedAlpha = 1.0},
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){
            .testName = "stiffened_gas_plus_stiffened_gas",
            .eosGas = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .dim = 3,
            .conservedValuesIn = {847.3900023809954, 1541208988.157894, 8473.900023809954, -16947.80004761991, 25421.700071429863, 499.45858996434845},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
            .normalIn = {0.5, 0.5, 0.7071},                                                                                                              // x, y, z
            .expectedDensity = 847.3900023809954,
            .expectedDensityG = 768.3978307143823,
            .expectedDensityL = 994.0897497618482,
            .expectedNormalVelocity = 16.213,
            .expectedVelocity = {10.0, -20.0, 30.0},
            .expectedInternalEnergy = 1818071.7389011043,
            .expectedInternalEnergyG = 1392890.3090808,
            .expectedInternalEnergyL = 2428423.405461103,
            .expectedSoundSpeedG = 1350.949940807393,
            .expectedSoundSpeedL = 1504.454840797822,
            .expectedMG = 0.012001184877590896,
            .expectedML = 0.010776661126898394,
            .expectedPressure = 100000.0,
            .expectedAlpha = 0.65},
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){.testName = "stiffened_gas_plus_stiffened_gas_all_water",
                                                          .eosGas = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                                                              {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
                                                          .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                                                              {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                                          .dim = 3,
                                                          .conservedValuesIn = {768.3978307143822, 1070293891.9207724, 0.0, 0.0, 0.0, 768.3978307143822},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
                                                          .normalIn = {0.5, 0.5, 0.7071},                                                                  // x, y, z
                                                          .expectedDensity = 768.3978307143822,
                                                          .expectedDensityG = 768.3978307143822,
                                                          .expectedDensityL = 994.0897497618488,
                                                          .expectedNormalVelocity = 0.0,
                                                          .expectedVelocity = {0.0, 0.0, 0.0},
                                                          .expectedInternalEnergy = 1392890.3090808005,
                                                          .expectedInternalEnergyG = 1392890.3090808005,
                                                          .expectedInternalEnergyL = 2428423.4054611027,
                                                          .expectedSoundSpeedG = 1350.9499408073941,
                                                          .expectedSoundSpeedL = 1504.4548407978223,
                                                          .expectedMG = 0.0,
                                                          .expectedML = 0.0,
                                                          .expectedPressure = 100000.0,
                                                          .expectedAlpha = 1.0},
        (TwoPhaseEulerAdvectionTestDecodeStateParameters){.testName = "stiffened_gas_plus_stiffened_gas_all_kerosene",
                                                          .eosGas = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                                                              {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
                                                          .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                                                              {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                                          .dim = 3,
                                                          .conservedValuesIn = {994.0897497618486, 2414070815.450644, 0.0, 0.0, 0.0, 0.0},  // RHO, RHOE, RHOU, RHOV, RHOW, RHOALPHA
                                                          .normalIn = {0.5, 0.5, 0.7071},                                                   // x, y, z
                                                          .expectedDensity = 994.0897497618486,
                                                          .expectedDensityG = 768.3978307143825,
                                                          .expectedDensityL = 994.0897497618486,
                                                          .expectedNormalVelocity = 0.0,
                                                          .expectedVelocity = {0.0, 0.0, 0.0},
                                                          .expectedInternalEnergy = 2428423.405461103,
                                                          .expectedInternalEnergyG = 1392890.3090808003,
                                                          .expectedInternalEnergyL = 2428423.405461103,
                                                          .expectedSoundSpeedG = 1350.9499408073932,
                                                          .expectedSoundSpeedL = 1504.454840797822,
                                                          .expectedMG = 0.0,
                                                          .expectedML = 0.0,
                                                          .expectedPressure = 100000.0,
                                                          .expectedAlpha = 0.0}),
    [](const testing::TestParamInfo<TwoPhaseEulerAdvectionTestDecodeStateParameters>& info) { return std::to_string(info.index); });

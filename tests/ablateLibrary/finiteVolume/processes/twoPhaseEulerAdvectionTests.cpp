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
    std::shared_ptr<eos::EOS> eosGas;
    std::shared_ptr<eos::EOS> eosLiquid;
    PetscInt dim;
    std::vector<PetscReal> conservedValuesIn;
    PetscReal densityVFIn;
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
    PetscReal alpha;

    // act
    FAIL() << "update";
    //    finiteVolume::processes::TwoPhaseEulerAdvection::DecodeTwoPhaseEulerState(
    //                                                                              params.dim,
    //                                                                              &params.conservedValuesIn[0],
    //                                                                              params.densityVFIn,
    //                                                                              &params.normalIn[0],
    //                                                                              &density,
    //                                                                              &densityG,
    //                                                                              &densityL,
    //                                                                              &normalVelocity,
    //                                                                              &velocity[0],
    //                                                                              &internalEnergy,
    //                                                                              &internalEnergyG,
    //                                                                              &internalEnergyL,
    //                                                                              &soundSpeedG,
    //                                                                              &soundSpeedL,
    //                                                                              &MG,
    //                                                                              &ML,
    //                                                                              &pressure,
    //                                                                              &alpha);

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

INSTANTIATE_TEST_SUITE_P(TwoPhaseEulerAdvectionTests, TwoPhaseEulerAdvectionTestDecodeStateFixture,
                         testing::Values(
                             (TwoPhaseEulerAdvectionTestDecodeStateParameters){
                                 // all phase 1, near boundary
                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                                 .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
                                 .dim = 3,
                                 .conservedValuesIn = {3.9, 936986.7, 39.0, -78.0, 117.0},  // RHO, RHOE, RHOU, RHOV, RHOW
                                 .densityVFIn = 3.9 - 1E-4,                                 // RHOALPHA
                                 .normalIn = {0.5, 0.5, 0.7071},                            // x, y, z
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
                                 // all phase 2, near boundary
                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                                 .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
                                 .dim = 3,
                                 .conservedValuesIn = {3.9, 936986.7, 39.0, -78.0, 117.0},  // RHO, RHOE, RHOU, RHOV, RHOW
                                 .densityVFIn = 0.0001,                                     // RHOALPHA
                                 .normalIn = {0.5, 0.5, 0.7071},                            // x, y, z
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
                                 // perfect gas + perfect gas
                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                                 .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
                                 .dim = 3,
                                 .conservedValuesIn = {3.9, 936986.7, 39.0, -78.0, 117.0},  // RHO, RHOE, RHOU, RHOV, RHOW
                                 .densityVFIn = 0.8,                                        // RHOALPHA
                                 .normalIn = {0.5, 0.5, 0.7071},                            // x, y, z
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
                                 // perfect gas + stiffened gas
                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                                 .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                                     {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                 .dim = 3,
                                 .conservedValuesIn = {340.4, 851000000.0, 3404.0, -6808.0, 10212.0},  // RHO, RHOE, RHOU, RHOV, RHOW
                                 .densityVFIn = 0.8,                                                   // RHOALPHA
                                 .normalIn = {0.5, 0.5, 0.7071},                                       // x, y, z
                                 .expectedDensity = 340.4,
                                 .expectedDensityG = 1.2352213315429057,
                                 .expectedDensityL = 963.834108739261,
                                 .expectedNormalVelocity = 16.213,
                                 .expectedVelocity = {10.0, -20.0, 30.0},
                                 .expectedInternalEnergy = 2499300.0,
                                 .expectedInternalEnergyG = 222008.74197557007,
                                 .expectedInternalEnergyL = 2504664.643717372,
                                 .expectedSoundSpeedG = 352.5973560682485,
                                 .expectedSoundSpeedL = 1527.8918538115954,
                                 .expectedMG = 0.04598162669393875,
                                 .expectedML = 0.01061135312656705,
                                 .expectedPressure = 109691.9735517502,
                                 .expectedAlpha = 0.6476572089266498},
                             (TwoPhaseEulerAdvectionTestDecodeStateParameters){
                                 // perfect gas + stiffened gas, all water
                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                                 .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                                     {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                 .dim = 3,
                                 .conservedValuesIn = {994.0897497618486, 2414070815.4506435, 0.0, 0.0, 0.0},  // RHO, RHOE, RHOU, RHOV, RHOW
                                 .densityVFIn = 0.0,                                                           // RHOALPHA
                                 .normalIn = {0.5, 0.5, 0.7071},                                               // x, y, z
                                 .expectedDensity = 994.0897497618486,
                                 .expectedDensityG = 1.161488313443748,
                                 .expectedDensityL = 994.0897497624699,
                                 .expectedNormalVelocity = 0.0,
                                 .expectedVelocity = {0.0, 0.0, 0.0},
                                 .expectedInternalEnergy = 2428423.4054611027,
                                 .expectedInternalEnergyG = 215241.08389368636,
                                 .expectedInternalEnergyL = 2428423.4054611027,
                                 .expectedSoundSpeedG = 347.18151877723034,
                                 .expectedSoundSpeedL = 1504.4548407982602,
                                 .expectedMG = 0.0,
                                 .expectedML = 0.0,
                                 .expectedPressure = 100000.0014061928,
                                 .expectedAlpha = 0.0},
                             (TwoPhaseEulerAdvectionTestDecodeStateParameters){
                                 // perfect gas + stiffened gas, all air
                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                                 .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
                                     {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                 .dim = 3,
                                 .conservedValuesIn = {1.1614401858304297, 250000.0, 0.0, 0.0, 0.0},  // RHO, RHOE, RHOU, RHOV, RHOW
                                 .densityVFIn = 1.1614401858304297,                                   // RHOALPHA
                                 .normalIn = {0.5, 0.5, 0.7071},                                      // x, y, z
                                 .expectedDensity = 1.1614401858304297,
                                 .expectedDensityG = 1.16144018583043,
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
                                 .expectedAlpha = 1.0}),
                         [](const testing::TestParamInfo<TwoPhaseEulerAdvectionTestDecodeStateParameters>& info) { return std::to_string(info.index); });

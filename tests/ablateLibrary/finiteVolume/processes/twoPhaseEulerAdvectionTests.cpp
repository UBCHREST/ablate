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
    finiteVolume::processes::TwoPhaseEulerAdvection::DecodeTwoPhaseEulerState(params.eosGas,
                                                                              params.eosLiquid,
                                                                              params.dim,
                                                                              &params.conservedValuesIn[0],
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
    ASSERT_NEAR(internalEnergyG, params.expectedInternalEnergyG, 1E-6);
    ASSERT_NEAR(internalEnergyL, params.expectedInternalEnergyL, 1E-6);
    ASSERT_NEAR(soundSpeedG, params.expectedSoundSpeedG, 1E-6);
    ASSERT_NEAR(soundSpeedL, params.expectedSoundSpeedL, 1E-6);
    ASSERT_NEAR(MG, params.expectedMG, 1E-6);
    ASSERT_NEAR(ML, params.expectedML, 1E-6);
    ASSERT_NEAR(pressure, params.expectedPressure, 1E-6);
    ASSERT_NEAR(alpha, params.expectedAlpha, 1E-6);
}

INSTANTIATE_TEST_SUITE_P(TwoPhaseEulerAdvectionTests, TwoPhaseEulerAdvectionTestDecodeStateFixture,
                         testing::Values(
//                         (TwoPhaseEulerAdvectionTestDecodeStateParameters){
//                                 // all phase 1
//                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
//                                 .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
//                                 .dim = 3,
//                                 .conservedValuesIn = {3.9-1E-5, 3.9, 936986.7, 39.0, -78.0, 117.0},  // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
//                                 .normalIn = {0.5, 0.5, 0.7071},                                 // x, y, z
//                                 .expectedDensity = 3.9,
//                                 .expectedDensityG = 3.9,
//                                 .expectedDensityL = 11.17065868263473,
//                                 .expectedNormalVelocity = 16.213,
//                                 .expectedVelocity = {10.0, -20.0, 30.0},
//                                 .expectedInternalEnergy = 239553.0,
//                                 .expectedInternalEnergyG = 239553.0,
//                                 .expectedInternalEnergyL = 15206.341843522327,
//                                 .expectedSoundSpeedG = 366.2644945937293,
//                                 .expectedSoundSpeedL = 327.1890074229225,
//                                 .expectedMG = 0.04426582494157374,
//                                 .expectedML = 0.04955239825353661,
//                                 .expectedAlpha = 1.0,
//                                 .expectedPressure = 373702.67999999993}),
                             (TwoPhaseEulerAdvectionTestDecodeStateParameters){
                                 // all phase 2
                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                                 .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
                                 .dim = 3,
                                 .conservedValuesIn = {1.04, 3.9, 936986.7, 39.0, -78.0, 117.0},  // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
                                 .normalIn = {0.5, 0.5, 0.7071},                                 // x, y, z
                                 .expectedDensity = 3.9,
                                 .expectedDensityG = 1.361602787456446,
                                 .expectedDensityL = 3.9,
                                 .expectedNormalVelocity = 16.213,
                                 .expectedVelocity = {10.0, -20.0, 30.0},
                                 .expectedInternalEnergy = 239553.0,
                                 .expectedInternalEnergyG = 3773796.511976049,
                                 .expectedInternalEnergyL = 239553.0,
                                 .expectedSoundSpeedG = 1453.728326306737,
                                 .expectedSoundSpeedL = 1298.6350988634183,
                                 .expectedMG = 0.011152702817031753,
                                 .expectedML = 0.01248464639080664,
                                 .expectedAlpha = 0.0,
                                 .expectedPressure = 2055364.74}),
//                             (TwoPhaseEulerAdvectionTestDecodeStateParameters){
//                                 // perfect gas + perfect gas
//                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
//                                 .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
//                                 .dim = 3,
//                                 .conservedValuesIn = {0.8, 3.9, 936986.7, 39.0, -78.0, 117.0},  // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
//                                 .normalIn = {0.5, 0.5, 0.7071},                                 // x, y, z
//                                 .expectedDensity = 3.9,
//                                 .expectedDensityG = 1.88229965,
//                                 .expectedDensityL = 5.391417167,
//                                 .expectedNormalVelocity = 16.213,
//                                 .expectedVelocity = {10.0, -20.0, 30.0},
//                                 .expectedInternalEnergy = 239553.0,
//                                 .expectedInternalEnergyG = 937273.0745446227,
//                                 .expectedInternalEnergyL = 59496.20656912966,
//                                 .expectedSoundSpeedG = 724.4811396751392,
//                                 .expectedSoundSpeedL = 647.1887624539481,
//                                 .expectedMG = 0.02237877442505955,
//                                 .expectedML = 0.0250514238512503,
//                                 .expectedAlpha = 0.425012032,
//                                 .expectedPressure = 705691.5126557435},
//                             (TwoPhaseEulerAdvectionTestDecodeStateParameters){
//                                 // perfect gas + stiffened gas
//                                 .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
//                                 .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
//                                     {"gamma", "2.4"}, {"Cv", "3030.0"}, {"p0", "1.0e7"}, {"T0", "584.25"}, {"e0", "1393000.0"}})),
//                                 .dim = 3,
//                                 .conservedValuesIn = {0.8, 3.9, 936986.7, 39.0, -78.0, 117.0},  // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
//                                 .normalIn = {0.5, 0.5, 0.7071},                                 // x, y, z
//                                 .expectedDensity = 3.9,
//                                 .expectedDensityG = 0.8397454729123517,
//                                 .expectedDensityL = 65.49704344363963,
//                                 .expectedNormalVelocity = 16.213,
//                                 .expectedVelocity = {10.0, -20.0, 30.0},
//                                 .expectedInternalEnergy = 239553.0,
//                                 .expectedInternalEnergyG = 151448.64669659876,
//                                 .expectedInternalEnergyL = 262289.60730410356,
//                                 .expectedSoundSpeedG = 291.22369778246974,
//                                 .expectedSoundSpeedL = 606.8713416892093,
//                                 .expectedMG = 0.05567198041730224,
//                                 .expectedML = 0.02671571202369117,
//                                 .expectedPressure = 50871.32617686838,
//                                 .expectedAlpha = 0.9526696193139227},
//                             (TwoPhaseEulerAdvectionTestDecodeStateParameters){
//                                 // stiffened gas + stiffened gas
//                                 .eosGas = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
//                                     {"gamma", "2.4"}, {"Cv", "3030.0"}, {"p0", "1.0e7"}, {"T0", "584.25"}, {"e0", "1393000.0"}})),
//                                 .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{
//                                     {"gamma", "3.2"}, {"Cv", "100.2"}, {"p0", "3.5e6"}, {"T0", "654.32"}, {"e0", "1482000.9"}})),
//                                 .dim = 3,
//                                 .conservedValuesIn = {0.8, 3.9, 9369867.0, 39.0, -78.0, 117.0},  // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
//                                 .normalIn = {0.5, 0.5, 0.7071},                                  // x, y, z
//                                 .expectedDensity = 3.9,
//                                 .expectedDensityG = 3.525420883761893,
//                                 .expectedDensityL = 4.00995119864821,
//                                 .expectedNormalVelocity = 16.213,
//                                 .expectedVelocity = {10.0, -20.0, 30.0},
//                                 .expectedInternalEnergy = 2401830.0,
//                                 .expectedInternalEnergyG = 5470826.91591592,
//                                 .expectedInternalEnergyL = 1609830.7958926656,
//                                 .expectedSoundSpeedG = 2975.0968917461205,
//                                 .expectedSoundSpeedL = 2277.8269509487473,
//                                 .expectedMG = 0.00544957041398554,
//                                 .expectedML = 0.007117748779487861,
//                                 .expectedPressure = 3001754.445143316,
//                                 .expectedAlpha = 0.22692326005238245}),
                         [](const testing::TestParamInfo<TwoPhaseEulerAdvectionTestDecodeStateParameters>& info) { return std::to_string(info.index); });

#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <cmath>
#include <flow/boundaryConditions/essentialGhost.hpp>
#include <memory>
#include <mesh/boxMesh.hpp>
#include <monitors/solutionErrorMonitor.hpp>
#include <vector>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "flow/boundaryConditions/ghost.hpp"
#include "flow/compressibleFlow.hpp"
#include "flow/processes/twoPhaseEulerAdvection.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
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
    flow::processes::TwoPhaseEulerAdvection::DecodeTwoPhaseEulerState(
        params.eosGas, params.eosLiquid, params.dim, &params.conservedValuesIn[0], &params.normalIn[0], &density, &densityG, &densityL, &normalVelocity, &velocity[0], &internalEnergy, &internalEnergyG, &internalEnergyL, &soundSpeedG, &soundSpeedL, &MG, &ML, &pressure, &alpha);

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
    testing::Values((TwoPhaseEulerAdvectionTestDecodeStateParameters){ // perfect gas + perfect gas
                        .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string,std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                        .eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string,std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
                        .dim = 3,
                        .conservedValuesIn = {0.8, 3.9, 936986.7, 39.0, -78.0, 117.0}, // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
                        .normalIn = {0.5, 0.5, 0.7071}, // x, y, z
                        .expectedDensity = 3.9,
                        .expectedDensityG = 1.88229965,
                        .expectedDensityL = 5.391417167,
                        .expectedNormalVelocity = 33.72128712,
                        .expectedVelocity = {10.0, -20.0, 30.0},
                        .expectedInternalEnergy = 239553.0,
                        .expectedInternalEnergyG = 937273.0745446227,
                        .expectedInternalEnergyL = 59496.20656912966,
                        .expectedSoundSpeedG = 724.4811396751392,
                        .expectedSoundSpeedL = 647.1887624539481,
                        .expectedMG = 0.04654543130704657,
                        .expectedML = 0.05210425315813406,
                        .expectedPressure = 705691.5126557435,
                        .expectedAlpha = 0.425012032}),
//                    (TwoPhaseEulerAdvectionTestDecodeStateParameters){ // perfect gas + stiffened gas
//                        .eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string,std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
//                        .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string,std::string>{{"gamma", "2.4"}, {"Cv", "3030.0"},{"p0","1.0e7"},{"T0","584.25"},{"e0","1393000.0"}})),
//                        .conservedValuesIn = {0.8, 3.9, 936986.7, 39.0, -78.0, 117.0}, // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
//                        .normalIn = {0.5, 0.5, 0.7071}, // x, y, z
//                        .expectedDensity = 3.9,
//                        .expectedDensityG = 0.8397454729123517,
//                        .expectedDensityL = 65.49704344363963,
//                        .expectedNormalVelocity = 33.72128712,
//                        .expectedVelocity = {10.0, -20.0, 30.0},
//                        .expectedInternalEnergy = 239553.0,
//                        .expectedInternalEnergyG = 151448.64669659876,
//                        .expectedInternalEnergyL = 262289.60730410356,
//                        .expectedSoundSpeedG = 291.22369778246974,
//                        .expectedSoundSpeedL = 606.8713416892093,
//                        .expectedMG = 0.11579170025232012,
//                        .expectedML = 0.055565792621114295,
//                        .expectedPressure = 50871.32617686838,
//                        .expectedAlpha = 0.9526696193139227},
//                    (TwoPhaseEulerAdvectionTestDecodeStateParameters){ // stiffened gas + stiffened gas
//                        .eosGas = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string,std::string>{{"gamma", "2.4"}, {"Cv", "3030.0"},{"p0","1.0e7"},{"T0","584.25"},{"e0","1393000.0"}})),
//                        .eosLiquid = std::make_shared<eos::StiffenedGas>(std::make_shared<parameters::MapParameters>(std::map<std::string,std::string>{{"gamma", "3.2"}, {"Cv", "100.2"},{"p0","3.5e6"},{"T0","654.32"},{"e0","1393000.0"}})),
//                        .conservedValuesIn = {0.5, 1.17, 3000.0, 0.0, 0.0, 0.0}, // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
//                        .normalIn = {1.0, 0.0, 0.0}, // x, y, z
//                        .expectedDensity = 1.2,
//                        .expectedDensityG = 1.2,
//                        .expectedDensityL = 1.2,
//                        .expectedNormalVelocity = 1.2,
//                        .expectedVelocity = {0, 0, 0},
//                        .expectedInternalEnergy = 99300,
//                        .expectedInternalEnergyG = 1.2,
//                        .expectedInternalEnergyL = 1.2,
//                        .expectedSoundSpeedG = 235.8134856,
//                        .expectedSoundSpeedL = 1.2,
//                        .expectedMG = 1.2,
//                        .expectedML = 1.2,
//                        .expectedPressure = 47664,
//                        .expectedAlpha = 0.5}),
    [](const testing::TestParamInfo<TwoPhaseEulerAdvectionTestDecodeStateParameters>& info) { return std::to_string(info.index); });

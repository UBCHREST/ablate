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
#include "flow/boundaryConditions/ghost.hpp"
#include "flow/compressibleFlow.hpp"
#include "flow/processes/twoPhaseEulerAdvection.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "parameters/mapParameters.hpp"

using namespace ablate;

struct TwoPhaseEulerAdvectionTestDecodeStateParameters {
    std::map<std::string, std::string> options; // what is this?
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
auto parameters = std::make_shared<ablate::parameters::MapParameters>(GetParam().options);
//std::shared_ptr<ablate::eos::EOS> eosGas = std::make_shared<ablate::eos::PerfectGas>(parameters);
//std::shared_ptr<ablate::eos::EOS> eosLiquid = std::make_shared<ablate::eos::PerfectGas>(parameters);

// get the test params
const auto& params = GetParam();

// Prepare outputs
PetscReal density;
PetscReal densityG;
PetscReal densityL;
PetscReal normalVelocity;
std::vector<PetscReal> velocity;
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
PetscErrorCode ierr = flow::processes::TwoPhaseEulerAdvection::DecodeTwoPhaseEulerState(
    params.eosGas, params.eosLiquid, params.dim, &params.conservedValuesIn[0], &params.normalIn[0], &density, &densityG, &densityL, &normalVelocity, &velocity[0], &internalEnergy, &internalEnergyG, &internalEnergyL, &soundSpeedG, &soundSpeedL, &MG, &ML, &pressure, &alpha);

// assert
ASSERT_EQ(ierr, 0);
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
    testing::Values((TwoPhaseEulerAdvectionTestDecodeStateParameters){.options = {{"gamma", "1.4"}, {"Rgas", "287.0"}},
                        .conservedValuesIn = {0.5, 1.17, 3000.0, 0.0, 0.0, 0.0}, // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
                        .normalIn = {1.0, 0.0, 0.0}, // x, y, z
                        .expectedDensity = 1.2,
                        .expectedDensityG = 1.2,
                        .expectedDensityL = 1.2,
                        .expectedNormalVelocity = 1.2,
                        .expectedVelocity = {0, 0, 0},
                        .expectedInternalEnergy = 99300,
                        .expectedInternalEnergyG = 1.2,
                        .expectedInternalEnergyL = 1.2,
                        .expectedSoundSpeedG = 235.8134856,
                        .expectedSoundSpeedL = 1.2,
                        .expectedMG = 1.2,
                        .expectedML = 1.2,
                        .expectedPressure = 47664,
                        .expectedAlpha = 0.5},
                    (TwoPhaseEulerAdvectionTestDecodeStateParameters){.options = {{"gamma", "2.0"}, {"Rgas", "4.0"}},
                        .conservedValuesIn = {0.5, 1.17, 3000.0, 0.0, 0.0, 0.0}, // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
                        .normalIn = {1.0, 0.0, 0.0}, // x, y, z
                        .expectedDensity = 1.2,
                        .expectedDensityG = 1.2,
                        .expectedDensityL = 1.2,
                        .expectedNormalVelocity = 1.2,
                        .expectedVelocity = {0, 0, 0},
                        .expectedInternalEnergy = 99300,
                        .expectedInternalEnergyG = 1.2,
                        .expectedInternalEnergyL = 1.2,
                        .expectedSoundSpeedG = 235.8134856,
                        .expectedSoundSpeedL = 1.2,
                        .expectedMG = 1.2,
                        .expectedML = 1.2,
                        .expectedPressure = 47664,
                        .expectedAlpha = 0.5}),
[](const testing::TestParamInfo<TwoPhaseEulerAdvectionTestDecodeStateParameters>& info) { return std::to_string(info.index); });

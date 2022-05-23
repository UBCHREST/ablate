#include <eos/tChemV1.hpp>
#include <finiteVolume/fieldFunctions/compressibleFlowState.hpp>
#include <finiteVolume/fieldFunctions/massFractions.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include <mathFunctions/functionFactory.hpp>
#include <mathFunctions/mathFunction.hpp>
#include <vector>
#include "PetscTestFixture.hpp"
#include "eos/perfectGas.hpp"
#include "functional"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

using namespace ablate;

struct CompressibleFlowStateTestParameters {
    std::vector<PetscReal> location;
    std::function<std::shared_ptr<ablate::eos::EOS>()> eosFunction;
    std::shared_ptr<mathFunctions::MathFunction> temperature;
    std::shared_ptr<mathFunctions::MathFunction> pressure;
    std::shared_ptr<mathFunctions::MathFunction> velocity;
    std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> massFractions;
    std::vector<PetscReal> expectedEuler;
    std::vector<PetscReal> expectedDensityYi;
};

class CompressibleFlowStateTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<CompressibleFlowStateTestParameters> {};

TEST_P(CompressibleFlowStateTestFixture, ShouldComputeCorrectEuler) {
    // arrange
    const auto& params = GetParam();
    auto eos = params.eosFunction();

    auto massFractionFieldFunction = std::make_shared<ablate::finiteVolume::fieldFunctions::MassFractions>(eos, params.massFractions);
    finiteVolume::fieldFunctions::CompressibleFlowState flowState(eos, params.temperature, params.pressure, params.velocity, massFractionFieldFunction);
    const auto& location = params.location;

    auto computeStateFunction = flowState.GetFieldFunction("euler");

    // act
    std::vector<PetscScalar> eulerCompute(params.expectedEuler.size());
    PetscErrorCode ierr = computeStateFunction->GetPetscFunction()(location.size(), 0.0, &location[0], params.expectedEuler.size(), &eulerCompute[0], computeStateFunction->GetContext());

    // assert
    ASSERT_EQ(ierr, 0);
    for (std::size_t i = 0; i < eulerCompute.size(); i++) {
        ASSERT_NEAR(eulerCompute[i], params.expectedEuler[i], 1E-3);
    }
}

TEST_P(CompressibleFlowStateTestFixture, ShouldComputeCorrectMassFractions) {
    // arrange
    const auto& params = GetParam();
    auto eos = params.eosFunction();
    auto massFractionFieldFunction = std::make_shared<ablate::finiteVolume::fieldFunctions::MassFractions>(eos, params.massFractions);
    finiteVolume::fieldFunctions::CompressibleFlowState flowState(eos, params.temperature, params.pressure, params.velocity, massFractionFieldFunction);
    const auto& location = params.location;
    auto computeStateFunction = flowState.GetFieldFunction("densityYi");

    // act
    std::vector<PetscScalar> densityYi(eos->GetSpecies().size());
    PetscErrorCode ierr = computeStateFunction->GetPetscFunction()(location.size(), 0.0, &location[0], params.expectedEuler.size(), &densityYi[0], computeStateFunction->GetContext());

    // assert
    ASSERT_EQ(ierr, 0);
    for (std::size_t i = 0; i < densityYi.size(); i++) {
        ASSERT_NEAR(densityYi[i], params.expectedDensityYi[i], 1E-3);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FieldFunctionTests, CompressibleFlowStateTestFixture,
    testing::Values((CompressibleFlowStateTestParameters){.location = {0.0, 0.0, 0.0},
                                                          .eosFunction = []() { return std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>()); },
                                                          .temperature = mathFunctions::Create(138.3972),
                                                          .pressure = mathFunctions::Create(47664.0),
                                                          .velocity = mathFunctions::Create(std::vector<double>{10, -20, 30}),
                                                          .massFractions = {},
                                                          .expectedEuler = {1.2, 1.2 * 1E5, 1.2 * 10, 1.2 * -20.0, 1.2 * 30},
                                                          .expectedDensityYi = {}},
                    (CompressibleFlowStateTestParameters){.location = {0.0, 0.0, 0.0},
                                                          .eosFunction =
                                                              []() {
                                                                  return std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(), std::vector<std::string>{"N2", "H2O", "CO2"});
                                                              },
                                                          .temperature = mathFunctions::Create(138.3972),
                                                          .pressure = mathFunctions::Create(47664.0),
                                                          .velocity = mathFunctions::Create(std::vector<double>{10, -20, 30}),
                                                          .massFractions =
                                                              {
                                                                  std::make_shared<ablate::mathFunctions::FieldFunction>("CO2", mathFunctions::Create(0.2)),
                                                                  std::make_shared<ablate::mathFunctions::FieldFunction>("N2", mathFunctions::Create(0.8)),
                                                              },
                                                          .expectedEuler = {1.2, 1.2 * 1E5, 1.2 * 10, 1.2 * -20.0, 1.2 * 30},
                                                          .expectedDensityYi = {1.2 * 0.8, 1.2 * 0.0, 1.2 * 0.2}},
                    (CompressibleFlowStateTestParameters){
                        .location = {0.0},
                        .eosFunction =
                            []() {
                                return std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "2.0"}, {"Rgas", "4.0"}}));
                            },
                        .temperature = mathFunctions::Create(39000.0),
                        .pressure = mathFunctions::Create(140400.0),
                        .velocity = mathFunctions::Create(std::vector<double>{0.0}),
                        .massFractions = {},
                        .expectedEuler = {.9, .9 * 1.56E5, 0},
                        .expectedDensityYi = {}},
                    (CompressibleFlowStateTestParameters){.location = {0.0, 0.0, 0.0},
                                                          .eosFunction = []() { return std::make_shared<eos::TChemV1>("inputs/eos/grimech30.dat", "inputs/eos/thermo30.dat"); },
                                                          .temperature = mathFunctions::Create(499.25),
                                                          .pressure = mathFunctions::Create(197710.5),
                                                          .velocity = mathFunctions::Create(std::vector<double>{10, -20, 30}),
                                                          .massFractions =
                                                              {
                                                                  std::make_shared<ablate::mathFunctions::FieldFunction>("CH4", mathFunctions::Create(0.2)),
                                                                  std::make_shared<ablate::mathFunctions::FieldFunction>("O2", mathFunctions::Create(0.3)),
                                                                  std::make_shared<ablate::mathFunctions::FieldFunction>("N2", mathFunctions::Create(0.5)),
                                                              },
                                                          .expectedEuler = {1.2, 119992.792036, 1.2 * 10, -1.2 * 20, 1.2 * 30},
                                                          .expectedDensityYi = {0, 0, 0, 1.2 * 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.2 * 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        0,
                                                                                0, 0, 0, 0,         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.2 * 0.5}},
                    (CompressibleFlowStateTestParameters){.location = {0.0, 0.0, 0.0},
                                                          .eosFunction = []() { return std::make_shared<eos::TChemV1>("inputs/eos/grimech30.dat", "inputs/eos/thermo30.dat"); },
                                                          .temperature = mathFunctions::Create(762.664),
                                                          .pressure = mathFunctions::Create(189973.54),
                                                          .velocity = mathFunctions::Create(std::vector<double>{0.0, 0.0, 0.0}),
                                                          .massFractions = {std::make_shared<ablate::mathFunctions::FieldFunction>("O2", mathFunctions::Create(0.3)),
                                                                            std::make_shared<ablate::mathFunctions::FieldFunction>("N2", mathFunctions::Create(0.4)),
                                                                            std::make_shared<ablate::mathFunctions::FieldFunction>("CH2", mathFunctions::Create(0.1)),
                                                                            std::make_shared<ablate::mathFunctions::FieldFunction>("NO", mathFunctions::Create(0.2))},
                                                          .expectedEuler = {0.8, 255999.11099, 0.0, 0.0, 0.0},
                                                          .expectedDensityYi = {0, 0, 0, 0.8 * 0.3, 0, 0, 0, 0, 0,         0, 0.8 * 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        0,
                                                                                0, 0, 0, 0,         0, 0, 0, 0, 0.8 * 0.2, 0, 0,         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8 * 0.4}}),
    [](const testing::TestParamInfo<CompressibleFlowStateTestParameters>& info) { return std::to_string(info.index); });

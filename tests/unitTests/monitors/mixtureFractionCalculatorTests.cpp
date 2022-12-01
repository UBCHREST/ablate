#include "PetscTestFixture.hpp"
#include "monitors/mixtureFractionCalculator.hpp"
#include "eos/perfectGas.hpp"
#include "eos/tChem.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/constantValue.hpp"
#include "parameters/mapParameters.hpp"
#include "utilities/vectorUtilities.hpp"

struct MixtureFractionCalculatorParameters {
    // inputs
    std::function<std::shared_ptr<ablate::eos::TChem>()> createEOS;
    std::map<std::string, double> massFractionsFuel;
    std::map<std::string, double> massFractionsOxidizer;
    std::vector<std::string> trackingElements;

    // test parameters
    std::vector<std::pair<std::map<std::string, double>, double>> parameters;
};

class MixtureFractionCalculatorFixture : public testing::TestWithParam<MixtureFractionCalculatorParameters> {};

TEST_P(MixtureFractionCalculatorFixture, ShouldComputeMixtureFraction) {
    // arrange
    auto eos = GetParam().createEOS();
    ablate::monitors::MixtureFractionCalculator mixtureFractionCalculator(eos, GetParam().massFractionsFuel, GetParam().massFractionsOxidizer, GetParam().trackingElements);

    // test each case
    for (const auto& [inputMassFractions, expectedValue] : GetParam().parameters) {
        // build a mixture fraction vector
        std::vector<double> mixtureFraction(eos->GetSpecies().size());
        for (const auto& [species, yi] : inputMassFractions) {
            auto location = std::find(eos->GetSpecies().begin(), eos->GetSpecies().end(), species);
            if (location != eos->GetSpecies().end()) {
                auto i = std::distance(eos->GetSpecies().begin(), location);
                mixtureFraction[i] = yi;
            }
        }
        // act
        auto zMix = mixtureFractionCalculator.Calculate(mixtureFraction.data());

        // assert
        ASSERT_NEAR(expectedValue, zMix, 1E-6);
    }
}

TEST_P(MixtureFractionCalculatorFixture, ShouldComputeMixtureFractionUsingFieldFunction) {
    // arrange
    auto eos = GetParam().createEOS();
    ablate::monitors::MixtureFractionCalculator mixtureFractionCalculator(
        eos,
        std::make_shared<ablate::mathFunctions::FieldFunction>(
            "yi", std::make_shared<ablate::mathFunctions::ConstantValue>(ablate::utilities::VectorUtilities::Fill(eos->GetSpecies(), GetParam().massFractionsFuel))),
        std::make_shared<ablate::mathFunctions::FieldFunction>(
            "yi", std::make_shared<ablate::mathFunctions::ConstantValue>(ablate::utilities::VectorUtilities::Fill(eos->GetSpecies(), GetParam().massFractionsOxidizer))),
        GetParam().trackingElements);

    // test each case
    for (const auto& [inputMassFractions, expectedValue] : GetParam().parameters) {
        // build a mixture fraction vector
        std::vector<double> mixtureFraction(eos->GetSpecies().size());
        for (const auto& [species, yi] : inputMassFractions) {
            auto location = std::find(eos->GetSpecies().begin(), eos->GetSpecies().end(), species);
            if (location != eos->GetSpecies().end()) {
                auto i = std::distance(eos->GetSpecies().begin(), location);
                mixtureFraction[i] = yi;
            }
        }
        // act
        auto zMix = mixtureFractionCalculator.Calculate(mixtureFraction.data());

        // assert
        ASSERT_NEAR(expectedValue, zMix, 1E-6);
    }
}

INSTANTIATE_TEST_SUITE_P(MixtureFractionCalculatorTests, MixtureFractionCalculatorFixture,
                         testing::Values((MixtureFractionCalculatorParameters){.createEOS =
                                                                                   []() { return std::make_shared<ablate::eos::TChem>("inputs/eos/grimech30.dat", "inputs/eos/thermo30.dat"); },
                                                                               .massFractionsFuel = {{"CH4", 1.0}},
                                                                               .massFractionsOxidizer = {{"O2", 1.0}},
                                                                               .parameters =
                                                                                   {
                                                                                       {{{"CH4", 1}}, 1.0},
                                                                                       {{{"O2", 1}}, 0.0},
                                                                                       {{{"CH4", .25}, {"O2", 0.75}}, 0.25},
                                                                                       {{{"CH4", .75}, {"O2", 0.25}}, 0.75},
                                                                                       {{{"CH4", .5}, {"O2", .2}, {"H2O", .3}}, 0.5335695032217096},
                                                                                       {{{"CH4", .1}, {"O2", .2}, {"H2O", .1}, {"AR", .6}}, 0.11118983440723654},
                                                                                   }},
                                         (MixtureFractionCalculatorParameters){.createEOS = []() { return std::make_shared<ablate::eos::TChem>("inputs/eos/gri30.yaml"); },
                                                                               .massFractionsFuel = {{"CH4", 0.7}, {"CH", 0.3}},
                                                                               .massFractionsOxidizer = {{"N2", 0.75511}, {"O2", 0.2314}, {"AR", 0.0129}, {"CO2", 0.00059}},
                                                                               .parameters = {
                                                                                   {{{"CH4", 0.7}, {"CH", 0.3}}, 1.0},
                                                                                   {{{"N2", 0.75511}, {"O2", 0.2314}, {"AR", 0.0129}, {"CO2", 0.00059}}, 0},
                                                                                   {{{"CH4", 0.23}, {"CH", 0.43}, {"AR", 0.0129}, {"CO2", 0.1}, {"O2", 0.1271}, {"N2", 0.2}}, 0.6872407932461698},
                                                                               }}));

struct MixtureFractionCalculatorExceptionParameters {
    std::string name;
    std::function<std::shared_ptr<ablate::eos::EOS>()> createEOS;
    std::map<std::string, double> massFractionsFuel;
    std::map<std::string, double> massFractionsOxidizer;
    std::vector<std::string> trackingElements;
};
class MixtureFractionCalculatorExceptionFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<MixtureFractionCalculatorExceptionParameters> {};

TEST_P(MixtureFractionCalculatorExceptionFixture, ShouldThrowExceptionWithInvalidInputs) {
    // arrange
    auto eos = GetParam().createEOS();
    ASSERT_THROW(ablate::monitors::MixtureFractionCalculator mixtureFractionCalculator(eos, GetParam().massFractionsFuel, GetParam().massFractionsOxidizer, GetParam().trackingElements);
                 , std::invalid_argument);
}

TEST_P(MixtureFractionCalculatorExceptionFixture, ShouldThrowExceptionWithInvalidInputsUsingFieldFunction) {
    // arrange
    auto eos = GetParam().createEOS();
    ASSERT_THROW(ablate::monitors::MixtureFractionCalculator mixtureFractionCalculator(
                     eos,
                     std::make_shared<ablate::mathFunctions::FieldFunction>(
                         "yi", std::make_shared<ablate::mathFunctions::ConstantValue>(ablate::utilities::VectorUtilities::Fill(eos->GetSpecies(), GetParam().massFractionsFuel))),
                     std::make_shared<ablate::mathFunctions::FieldFunction>(
                         "yi", std::make_shared<ablate::mathFunctions::ConstantValue>(ablate::utilities::VectorUtilities::Fill(eos->GetSpecies(), GetParam().massFractionsOxidizer))),
                     GetParam().trackingElements);
                 , std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(MixtureFractionCalculatorTests, MixtureFractionCalculatorExceptionFixture,
                         testing::Values((MixtureFractionCalculatorExceptionParameters){.name = "invalid EOS",
                                                                                        .createEOS =
                                                                                            []() { return std::make_shared<ablate::eos::PerfectGas>(ablate::parameters::MapParameters::Create({})); },
                                                                                        .massFractionsFuel = {{"CH4", 1.0}},
                                                                                        .massFractionsOxidizer = {{"O2", 1.0}}},
                                         (MixtureFractionCalculatorExceptionParameters){.name = "invalid massFractionsFuel",
                                                                                        .createEOS = []() { return std::make_shared<ablate::eos::TChem>("inputs/eos/gri30.yaml"); },
                                                                                        .massFractionsFuel = {{"CH4", 0.7}},
                                                                                        .massFractionsOxidizer = {{"N2", 0.75511}, {"O2", 0.2314}, {"AR", 0.0129}, {"CO2", 0.00059}}},
                                         (MixtureFractionCalculatorExceptionParameters){.name = "invalid massFractionsOxidizer",
                                                                                        .createEOS = []() { return std::make_shared<ablate::eos::TChem>("inputs/eos/gri30.yaml"); },
                                                                                        .massFractionsFuel = {{"CH4", 1.0}},
                                                                                        .massFractionsOxidizer = {{"N2", 0.75511}, {"O2", 0.2314}}}),
                         [](const testing::TestParamInfo<MixtureFractionCalculatorExceptionParameters>& info) { return testingResources::PetscTestFixture::SanitizeTestName(info.param.name); });
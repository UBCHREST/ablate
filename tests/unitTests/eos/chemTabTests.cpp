#include <yaml-cpp/yaml.h>
#include "PetscTestFixture.hpp"
#include "eos/chemTab.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "localPath.hpp"
#include "mockFactory.hpp"

#ifndef WITH_TENSORFLOW
#define ONLY_WITH_TENSORFLOW_CHECK                                       \
    SUCCEED() << ("Test is only applicable when built with TensorFlow"); \
    return;
#define ONLY_WITHOUT_TENSORFLOW_CHECK \
    {}
#else
#define ONLY_WITH_TENSORFLOW_CHECK \
    {}
#define ONLY_WITHOUT_TENSORFLOW_CHECK                                     \
    SUCCEED() << "Test is only applicable when built without TensorFlow"; \
    return;
#endif

/*******************************************************************************************************
 * This test ensure that the chemTabModel can be created using the input file
 */
TEST(ChemTabTests, ShouldCreateFromRegistar) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    std::shared_ptr<cppParserTesting::MockFactory> mockFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedClassType = "ablate::eos::ChemTab";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    auto mockSubFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedSubClassType = "";
    EXPECT_CALL(*mockSubFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedSubClassType));
    EXPECT_CALL(*mockSubFactory, Get(cppParser::ArgumentIdentifier<std::string>{})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("inputs/eos/chemTabTestModel_1"));
    EXPECT_CALL(*mockFactory, GetFactory("path")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(mockSubFactory));

    // act
    auto createMethod = Creator<ablate::eos::ChemistryModel>::GetCreateMethod(mockFactory->GetClassType());
    auto instance = createMethod(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the ablate::chemistry::ChemTabModel";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::eos::ChemTab>(instance) != nullptr) << " should be an instance of ablate::chemistry::ChemTabModel";
}

/*******************************************************************************************************
 * Tests for expected input/outputs
 */
struct ChemTabModelTestParameters {
    std::string modelPath;
    std::string testTargetFile;
};
class ChemTabModelTestFixture : public testingResources::PetscTestFixture, public testing::WithParamInterface<ChemTabModelTestParameters> {
   protected:
    YAML::Node testTargets;

    void SetUp() override {
        testingResources::PetscTestFixture::SetUp();
        testTargets = YAML::LoadFile(GetParam().testTargetFile);

        // this should be an array
        if (!testTargets.IsSequence()) {
            FAIL() << "The provided test targets " + GetParam().testTargetFile + " must be an sequence.";
        }
    }
};

TEST_P(ChemTabModelTestFixture, ShouldReturnCorrectSpeciesAndVariables) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // iterate over each test
    for (const auto& testTarget : testTargets) {
        // arrange
        ablate::eos::ChemTab chemTabModel(GetParam().modelPath);

        // act
        auto actualSpeciesVariables = chemTabModel.GetSpeciesVariables();
        auto actualSpecies = chemTabModel.GetSpecies();
        auto actualProgressVariables = chemTabModel.GetProgressVariables();

        // assert
        EXPECT_TRUE(actualSpeciesVariables.empty()) << "should report no transport species" << testTarget["testName"].as<std::string>();
        EXPECT_EQ(testTarget["species_names"].as<std::vector<std::string>>(), actualSpecies) << "should compute correct species name for model " << testTarget["testName"].as<std::string>();
        EXPECT_EQ(testTarget["cpv_names"].as<std::vector<std::string>>(), actualProgressVariables) << "should compute correct cpv names for model " << testTarget["testName"].as<std::string>();
        EXPECT_EQ(testTarget["species_names"].as<std::vector<std::string>>().size(), actualSpecies.size())
            << "should compute correct species name for model " << testTarget["testName"].as<std::string>();
    }
}

#define assert_float_close(expected, actual) EXPECT_NEAR(expected, actual, PetscAbs(5.0E-6 * actual))  // gives you relative error check

/*******************************************************************************************************
 * Tests for getting the Compute Mass Fractions Functions
 */
TEST_P(ChemTabModelTestFixture, ShouldComputeCorrectMassFractions) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // iterate over each test
    for (const auto& testTarget : testTargets) {
        // arrange
        ablate::eos::ChemTab chemTabModel(GetParam().modelPath);
        auto expectedMassFractions = testTarget["output_mass_fractions"].as<std::vector<double>>();
        auto inputProgressVariables = testTarget["input_cpvs"].as<std::vector<double>>();

        // act
        std::vector<PetscReal> actual(expectedMassFractions.size());
        chemTabModel.ComputeMassFractions(inputProgressVariables.data(), inputProgressVariables.size(), actual.data(), actual.size());

        // assert
        for (std::size_t r = 0; r < actual.size(); r++) {
            assert_float_close(expectedMassFractions[r], actual[r]) << "The value for [" << r << "] is incorrect for model " << testTarget["testName"].as<std::string>();
        }
    }
}

/*******************************************************************************************************
 * Tests for getting the Source and Source Energy Predictions
 */
TEST_P(ChemTabModelTestFixture, ShouldComputeCorrectSource) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // iterate over each test
    for (const auto& testTarget : testTargets) {
        // arrange
        ablate::eos::ChemTab chemTabModel(GetParam().modelPath);
        auto inputProgressVariables = testTarget["input_cpvs"].as<std::vector<double>>();
        auto expectedSourceEnergy = testTarget["output_source_energy"].as<double>();
        auto expectedSourceProgress = testTarget["output_source_terms"].as<std::vector<double>>();

        // assume a density
        PetscReal density = 1.5;

        // size up and set the expected input
        std::vector<PetscReal> conservedProgressVariable(chemTabModel.GetProgressVariables().size(), 0.0);
        for (std::size_t p = 0; p < inputProgressVariables.size(); p++) {
            conservedProgressVariable[p] = inputProgressVariables[p] * density;
        }

        // act
        // Size up the results based upon expected
        std::vector<PetscReal> actualSourceProgress(conservedProgressVariable.size(), 0.0);
        PetscReal actualSourceEnergy = 0.0;
        chemTabModel.ChemistrySource(density, conservedProgressVariable.data(), &actualSourceEnergy, actualSourceProgress.data());

        assert_float_close(expectedSourceEnergy, actualSourceEnergy) << "The sourceEnergy is incorrect for model " << testTarget["testName"].as<std::string>();

        for (std::size_t r = 0; r < expectedSourceProgress.size(); r++) {
            assert_float_close(expectedSourceProgress[r], actualSourceProgress[r]) << " the percent difference of (" << expectedSourceProgress[r] << ", " << actualSourceProgress[r]
                                                                                   << ") should be less than 5.0E-6 for index [" << r << "] for model " << testTarget["testName"].as<std::string>();
        }
    }
}

TEST_P(ChemTabModelTestFixture, ShouldComputeCorrectThermalProperties) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // iterate over each test
    for (const auto& testTarget : testTargets) {
        // ARRANGE
        auto chemTab = std::make_shared<ablate::eos::ChemTab>(GetParam().modelPath);
        auto expectedMassFractions = testTarget["output_mass_fractions"].as<std::vector<double>>();
        auto inputProgressVariables = testTarget["input_cpvs"].as<std::vector<double>>();
        auto inputProgressVariablesNames = testTarget["cpv_names"].as<std::vector<std::string>>();

        // build a new reference eos
        auto metadata = YAML::LoadFile(std::filesystem::path(GetParam().modelPath) / "metadata.yaml");
        auto tchem = std::make_shared<ablate::eos::TChem>(std::filesystem::path(GetParam().modelPath) / metadata["mechanism"].as<std::string>());

        // assume values for density and energy
        double density = 1.2;
        double densityEnergy = 1.2 * 1.0E+05;
        double momentum = 1.2 * 10;

        // build a conserved array for only euler
        std::vector<PetscReal> eulerConserved = {0.0, density, densityEnergy, momentum};
        std::vector<PetscReal> allFieldsConserved = eulerConserved;
        for (auto pv : inputProgressVariables) {
            allFieldsConserved.push_back(pv * density);
        }

        // create fake fields for testings
        auto fields = {ablate::domain::Field{.name = ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD, .numberComponents = 3, .offset = 1},
                       ablate::domain::Field{.name = ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD,
                                             .numberComponents = (PetscInt)inputProgressVariablesNames.size(),
                                             .components = inputProgressVariablesNames,
                                             .offset = (PetscInt)eulerConserved.size()}};

        // compute the reference temperature for other calculations
        auto tChemTemperatureFunction = tchem->GetThermodynamicMassFractionFunction(ablate::eos::ThermodynamicProperty::Temperature, fields);
        PetscReal tChemComputedTemperature;
        ASSERT_EQ(tChemTemperatureFunction.function(eulerConserved.data(), expectedMassFractions.data(), &tChemComputedTemperature, tChemTemperatureFunction.context.get()), 0);
        auto chemTabTemperatureFunction = chemTab->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, fields);
        PetscReal chemTabComputedTemperature;
        ASSERT_EQ(chemTabTemperatureFunction.function(allFieldsConserved.data(), &chemTabComputedTemperature, chemTabTemperatureFunction.context.get()), 0);
        ASSERT_FLOAT_EQ(chemTabComputedTemperature, tChemComputedTemperature) << " The TChem and ChemTab temperatures should be equal";

        // Now check the other thermodynamic properties
        auto testProperties = {ablate::eos::ThermodynamicProperty::Pressure,
                               ablate::eos::ThermodynamicProperty::Temperature,
                               ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                               ablate::eos::ThermodynamicProperty::SensibleEnthalpy,
                               ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume,
                               ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure,
                               ablate::eos::ThermodynamicProperty::SpeedOfSound,
                               ablate::eos::ThermodynamicProperty::Density};

        for (auto& testProperty : testProperties) {
            PetscReal tChemComputedProperty, chemTabComputedProperty;

            // Test the direction function
            auto tChemFunction = tchem->GetThermodynamicMassFractionFunction(testProperty, fields);
            ASSERT_EQ(tChemFunction.function(eulerConserved.data(), expectedMassFractions.data(), &tChemComputedProperty, tChemFunction.context.get()), 0);

            auto chemTabFunction = chemTab->GetThermodynamicFunction(testProperty, fields);
            ASSERT_EQ(chemTabFunction.function(allFieldsConserved.data(), &chemTabComputedProperty, chemTabFunction.context.get()), 0);

            ASSERT_FLOAT_EQ(tChemComputedProperty, chemTabComputedProperty) << " The TChem and ChemTab " << testProperty << " should be equal";

            // test the function where temperature is an input
            auto tChemFunctionTemperature = tchem->GetThermodynamicTemperatureMassFractionFunction(testProperty, fields);
            ASSERT_EQ(tChemFunctionTemperature.function(eulerConserved.data(), expectedMassFractions.data(), tChemComputedTemperature, &tChemComputedProperty, tChemFunctionTemperature.context.get()),
                      0);

            auto chemTabFunctionTemperature = chemTab->GetThermodynamicTemperatureFunction(testProperty, fields);
            ASSERT_EQ(chemTabFunctionTemperature.function(allFieldsConserved.data(), chemTabComputedTemperature, &chemTabComputedProperty, chemTabFunctionTemperature.context.get()), 0);

            ASSERT_FLOAT_EQ(tChemComputedProperty, chemTabComputedProperty) << " The TChem and ChemTab " << testProperty << " should be equal when computed with temperature";
        }
    }
}

/*******************************************************************************************************
 * Tests for getting the Progress Variables
 */
TEST_P(ChemTabModelTestFixture, ShouldComputeCorrectProgressVariables) {
    ONLY_WITH_TENSORFLOW_CHECK;

    for (const auto& testTarget : testTargets) {
        // arrange
        ablate::eos::ChemTab chemTabModel(GetParam().modelPath);
        auto expectedProgressVariables = testTarget["output_cpvs"].as<std::vector<double>>();
        auto inputMassFractions = testTarget["input_mass_fractions"].as<std::vector<double>>();

        // act
        // Size up the results based upon expected
        std::vector<PetscReal> actual(expectedProgressVariables.size());
        chemTabModel.ComputeProgressVariables(inputMassFractions.data(), inputMassFractions.size(), actual.data(), actual.size());

        // assert
        for (std::size_t r = 0; r < actual.size(); r++) {
            assert_float_close(expectedProgressVariables[r], actual[r]) << "The value for input set [" << r << "] is incorrect for model " << testTarget["testName"].as<std::string>();
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabTests, ChemTabModelTestFixture,
                         testing::Values((ChemTabModelTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1", .testTargetFile = "inputs/eos/chemTabTestModel_1/testTargets.yaml"}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// ChemTab FieldFunctionTests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ChemTabFieldFunctionTestParameters {
    // eos init
    std::filesystem::path modelPath;

    // field function init
    ablate::eos::ThermodynamicProperty property1;
    ablate::eos::ThermodynamicProperty property2;

    // inputs
    PetscReal property1Value;
    PetscReal property2Value;
    std::vector<PetscReal> velocity;
    std::map<std::string, PetscReal> yiMap;
    std::vector<PetscReal> expectedEulerValue;
};

/*
 * Helper function to fill mass fraction
 */
static std::vector<PetscReal> GetMassFraction(const std::vector<std::string>& species, const std::map<std::string, PetscReal>& yiIn) {
    std::vector<PetscReal> yi(species.size(), 0.0);

    for (const auto& value : yiIn) {
        // Get the index
        auto it = std::find(species.begin(), species.end(), value.first);
        if (it != species.end()) {
            auto index = std::distance(species.begin(), it);

            yi[index] = value.second;
        }
    }
    return yi;
}

class ChemTabFieldFunctionTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<ChemTabFieldFunctionTestParameters> {};

TEST_P(ChemTabFieldFunctionTestFixture, ShouldComputeField) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    auto chemTab = std::make_shared<ablate::eos::ChemTab>(GetParam().modelPath);

    // build a new reference eos
    auto metadata = YAML::LoadFile(std::filesystem::path(GetParam().modelPath) / "metadata.yaml");
    auto tchem = std::make_shared<ablate::eos::TChem>(std::filesystem::path(GetParam().modelPath) / metadata["mechanism"].as<std::string>());
    auto yi = GetMassFraction(tchem->GetSpecies(), GetParam().yiMap);

    // get the test params
    const auto& params = GetParam();
    std::vector<PetscReal> actualEulerValue(params.expectedEulerValue.size(), NAN);
    std::vector<PetscReal> actualDensityEvValue(chemTab->GetProgressVariables().size(), NAN);

    // compute the expected progress
    auto expectedEvValue = actualDensityEvValue;
    chemTab->ComputeProgressVariables(yi.data(), yi.size(), expectedEvValue.data(), expectedEvValue.size());

    // act
    auto stateEulerFunction = chemTab->GetFieldFunctionFunction(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD, params.property1, params.property2);
    stateEulerFunction(params.property1Value, params.property2Value, (PetscInt)params.velocity.size(), params.velocity.data(), yi.data(), actualEulerValue.data());
    auto stateDensityEvFunction = chemTab->GetFieldFunctionFunction(ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD, params.property1, params.property2);
    stateDensityEvFunction(params.property1Value, params.property2Value, (PetscInt)params.velocity.size(), params.velocity.data(), yi.data(), actualDensityEvValue.data());

    // assert
    for (std::size_t c = 0; c < params.expectedEulerValue.size(); c++) {
        ASSERT_LT(PetscAbs(params.expectedEulerValue[c] - actualEulerValue[c]) / (params.expectedEulerValue[c] + 1E-30), 1E-3)
            << "for component[" << c << "] of expectedEulerValue (" << params.expectedEulerValue[c] << " vs " << actualEulerValue[c] << ")";
    }
    for (std::size_t c = 0; c < expectedEvValue.size(); c++) {
        ASSERT_LT(PetscAbs(expectedEvValue[c] * params.expectedEulerValue[0] - actualDensityEvValue[c]) / (expectedEvValue[c] * params.expectedEulerValue[0] + 1E-30), 1E-3)
            << "for component[" << c << "] of densityEv_progress (" << yi[c] * params.expectedEulerValue[0] << " vs " << actualDensityEvValue[c] << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabTests, ChemTabFieldFunctionTestFixture,
                         testing::Values((ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property1Value = 499.25,
                                                                              .property2Value = 197710.5,
                                                                              .velocity = {10, 20, 30},
                                                                              .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                              .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}},
                                         (ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property1Value = 762.664,
                                                                              .property2Value = 189973.54,
                                                                              .velocity = {0.0, 0.0, 0.0},
                                                                              .yiMap = {{"O2", .3}, {"N2", .4}, {"CH2", .1}, {"NO", .2}},
                                                                              .expectedEulerValue = {0.8, 0.8 * 3.2E5, 0.0, 0.0, 0.0}},
                                         (ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property1Value = 418.079,
                                                                              .property2Value = 409488.10,
                                                                              .velocity = {0, 2, 4},
                                                                              .yiMap = {{"N2", 1.0}},
                                                                              .expectedEulerValue = {3.3, 3.3 * 1000, 0.0, 3.3 * 2, 3.3 * 4}},
                                         (ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                              .property1Value = 7411.11,
                                                                              .property2Value = 437.46,
                                                                              .velocity = {-1, -2, -3},
                                                                              .yiMap = {{"H2", .35}, {"H2O", .35}, {"N2", .3}},
                                                                              .expectedEulerValue = {0.01, 0.01 * 1E5, .01 * -1, .01 * -2, .01 * -3}},
                                         (ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                                                              .property1Value = 281125963.5,
                                                                              .property2Value = 394.59,
                                                                              .velocity = {-10, -20, -300},
                                                                              .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                              .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},
                                         (ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                              .property1Value = 281125963.5,
                                                                              .property2Value = -35256.942891550425,
                                                                              .velocity = {-10, -20, -300},
                                                                              .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                              .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},
                                         (ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property1Value = -35256.942891550425,
                                                                              .property2Value = 281125963.5,
                                                                              .velocity = {-10, -20, -300},
                                                                              .yiMap = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                              .expectedEulerValue = {999.9, 999.9 * 1E4, 999.9 * -10, 999.9 * -20, 999.9 * -300}},
                                         (ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property1Value = 99291.694615827029,
                                                                              .property2Value = 197710.5,
                                                                              .velocity = {10, 20, 30},
                                                                              .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                              .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}},
                                         (ChemTabFieldFunctionTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1",
                                                                              .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                                                              .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                                                              .property1Value = 197710.5,
                                                                              .property2Value = 99291.694615827029,
                                                                              .velocity = {10, 20, 30},
                                                                              .yiMap = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                              .expectedEulerValue = {1.2, 1.2 * 99993.99, 1.2 * 10, 1.2 * 20, 1.2 * 30}}

                                         ),

                         [](const testing::TestParamInfo<ChemTabFieldFunctionTestParameters>& info) {
                             return std::to_string(info.index) + "_from_" + std::string(to_string(info.param.property1)) + "_" + std::string(to_string(info.param.property2)) + "_with_" +
                                    info.param.modelPath.stem().string();
                         });

TEST_P(ChemTabModelTestFixture, ShouldComputeProgressVariablesMassFractionsInterchangeability) {
    GTEST_SKIP() << "Test is not working with ChemTab";
    ONLY_WITH_TENSORFLOW_CHECK;

    for (const auto& testTarget : testTargets) {
        // arrange
        auto chemTab = std::make_shared<ablate::eos::ChemTab>(GetParam().modelPath);
        auto expectedProgressVariables = testTarget["output_cpvs"].as<std::vector<double>>();
        auto inputMassFractions = testTarget["input_mass_fractions"].as<std::vector<double>>();

        // act
        // Size up the results based upon expected
        std::vector<PetscReal> actualProgressVariables(expectedProgressVariables.size());
        chemTab->ComputeProgressVariables(inputMassFractions.data(), inputMassFractions.size(), actualProgressVariables.data(), actualProgressVariables.size());

        std::vector<PetscReal> actualMassFractions(inputMassFractions.size());
        chemTab->ComputeMassFractions(actualProgressVariables.data(), actualProgressVariables.size(), actualMassFractions.data(), actualMassFractions.size());

        // assert
        for (std::size_t r = 0; r < actualProgressVariables.size(); r++) {
            assert_float_close(expectedProgressVariables[r], actualProgressVariables[r]) << "The value for input set [" << r << "] is incorrect for model " << testTarget["testName"].as<std::string>();
        }
        for (std::size_t r = 0; r < actualMassFractions.size(); r++) {
            assert_float_close(inputMassFractions[r], actualMassFractions[r]) << "The value for input mass fractions [" << r << "] is incorrect for model " << testTarget["testName"].as<std::string>();
        }
    }
}

/*********************************************************************************************************
 * Test for when tensorflow is not available
 */
TEST(ChemTabTests, ShouldReportTensorFlowLibraryMissing) {
    ONLY_WITHOUT_TENSORFLOW_CHECK;
    ASSERT_ANY_THROW(ablate::eos::ChemTab("inputs/eos/chemTabTestModel_1"));
}

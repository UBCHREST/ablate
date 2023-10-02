#include "chemTabTestFixture.hpp"
#include "domain/mockField.hpp"
#include "eos/chemTab.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"
#include "localPath.hpp"
#include "mockFactory.hpp"
#include "petscTestFixture.hpp"

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

class ChemTabModelRegistrarFixture : public testingResources::PetscTestFixture {};

/*******************************************************************************************************
 * This test ensure that the chemTabModel can be created using the input file
 */
TEST_F(ChemTabModelRegistrarFixture, ShouldCreateFromRegistrar) {
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

TEST_P(ChemTabTestFixture, ShouldReturnCorrectSpeciesAndVariables) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // iterate over each test
    for (const auto& testTarget : testTargets) {
        // arrange
        ablate::eos::ChemTab chemTabModel(GetParam().modelPath);

        // act
        auto actualSpeciesVariables = chemTabModel.GetSpeciesVariables();
        auto actualFieldFunction = chemTabModel.GetFieldFunctionProperties();
        auto actualProgressVariables = chemTabModel.GetProgressVariables();

        // assert
        EXPECT_TRUE(actualSpeciesVariables.empty()) << "should report no transport species" << testTarget["testName"].as<std::string>();
        EXPECT_EQ(testTarget["cpv_names"].as<std::vector<std::string>>(), actualFieldFunction) << "should compute correct cpv for model " << testTarget["testName"].as<std::string>();
        EXPECT_EQ(testTarget["cpv_names"].as<std::vector<std::string>>(), actualProgressVariables) << "should compute correct cpv names for model " << testTarget["testName"].as<std::string>();
    }
}

#define assert_float_close(expected, actual) EXPECT_NEAR(expected, actual, PetscAbs(5.0E-4 * actual))  // gives you relative error check

/*******************************************************************************************************
 * Tests for getting the Compute Mass Fractions Functions
 */
TEST_P(ChemTabTestFixture, ShouldComputeCorrectMassFractions) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // iterate over each test
    for (const auto& testTarget : testTargets) {
        // arrange
        ablate::eos::ChemTab chemTabModel(GetParam().modelPath);
        auto expectedMassFractions = testTarget["output_mass_fractions"].as<std::vector<double>>();
        auto inputProgressVariables = testTarget["input_cpvs"].as<std::vector<double>>();

        // act
        std::vector<PetscReal> actual(expectedMassFractions.size());
        chemTabModel.ComputeMassFractions(inputProgressVariables, actual);

        // assert
        for (std::size_t r = 0; r < actual.size(); r++) {
            assert_float_close(expectedMassFractions[r], actual[r]) << "The value for [" << r << "] is incorrect for model " << testTarget["testName"].as<std::string>();
        }
    }
}

/*******************************************************************************************************
 * Tests for getting the Source and Source Energy Predictions
 */
TEST_P(ChemTabTestFixture, ShouldComputeCorrectSource) {
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

        assert_float_close(expectedSourceEnergy * density, actualSourceEnergy) << "The sourceEnergy is incorrect for model " << testTarget["testName"].as<std::string>();

        for (std::size_t r = 0; r < expectedSourceProgress.size(); r++) {
            std::cerr << "expected source: " << expectedSourceProgress[r] << " actual source: " << actualSourceProgress[r] << std::endl << std::flush;
            assert_float_close(expectedSourceProgress[r] * density, actualSourceProgress[r])
                << " the percent difference of (" << expectedSourceProgress[r] << ", " << actualSourceProgress[r] << ") should be less than 5.0E-6 for index [" << r << "] for model "
                << testTarget["testName"].as<std::string>();
        }
    }
}

TEST_P(ChemTabTestFixture, ShouldComputeCorrectThermalProperties) {
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
        std::vector<PetscReal> chemTabFieldsConserved = eulerConserved;
        for (auto pv : inputProgressVariables) {
            chemTabFieldsConserved.push_back(pv * density);
        }
        chemTabFieldsConserved.resize(chemTabFieldsConserved.size() + expectedMassFractions.size());

        std::vector<PetscReal> tChemFieldsConserved = eulerConserved;
        for (auto yi : expectedMassFractions) {
            tChemFieldsConserved.push_back(yi * density);
        }

        // create fake fields for testings
        std::vector<ablate::domain::Field> chemTabFields = {
            ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD, 3, 1),
            ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD, inputProgressVariablesNames, (PetscInt)eulerConserved.size()),
            ablateTesting::domain::MockField::Create(
                ablate::eos::ChemTab::DENSITY_YI_DECODE_FIELD, tchem->GetSpeciesVariables(), (PetscInt)(eulerConserved.size() + inputProgressVariablesNames.size()))};
        std::vector<ablate::domain::Field> tChemFields = {
            ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD, 3, 1),
            ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD, tchem->GetSpeciesVariables(), (PetscInt)(eulerConserved.size()))};

        // compute the reference temperature for other calculations
        auto tChemTemperatureFunction = tchem->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, tChemFields);
        PetscReal tChemComputedTemperature;
        ASSERT_EQ(tChemTemperatureFunction.function(tChemFieldsConserved.data(), &tChemComputedTemperature, tChemTemperatureFunction.context.get()), 0);

        // Compute the mass fractions from chemTab
        auto chemTabDensityMassFractionDecode = chemTab->GetSolutionFieldUpdates().front();
        auto densityMassFractionFunctionDecode = std::get<0>(chemTabDensityMassFractionDecode);
        auto densityMassFractionFunctionContext = std::get<1>(chemTabDensityMassFractionDecode);
        PetscInt solutionOffsets[3] = {chemTabFields[0].offset, chemTabFields[1].offset, chemTabFields[2].offset};
        // Use the computed massFractions from chemTab
        ASSERT_EQ(densityMassFractionFunctionDecode(NAN, -1, nullptr, solutionOffsets, chemTabFieldsConserved.data(), densityMassFractionFunctionContext), 0);

        // make sure that the density mass fractions are correct
        for (std::size_t s = 0; s < expectedMassFractions.size(); ++s) {
            assert_float_close(chemTabFieldsConserved[s + chemTabFields[2].offset], tChemFieldsConserved[s + tChemFields[1].offset]) << "The density mass fraction should be correctly computed.";
        }

        // Compute the reference temperature
        auto chemTabTemperatureFunction = chemTab->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, chemTabFields);
        PetscReal chemTabComputedTemperature;
        ASSERT_EQ(chemTabTemperatureFunction.function(chemTabFieldsConserved.data(), &chemTabComputedTemperature, chemTabTemperatureFunction.context.get()), 0);
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
            auto tChemFunction = tchem->GetThermodynamicMassFractionFunction(testProperty, tChemFields);
            ASSERT_EQ(tChemFunction.function(tChemFieldsConserved.data(), expectedMassFractions.data(), &tChemComputedProperty, tChemFunction.context.get()), 0);

            auto chemTabFunction = chemTab->GetThermodynamicFunction(testProperty, chemTabFields);
            ASSERT_EQ(chemTabFunction.function(chemTabFieldsConserved.data(), &chemTabComputedProperty, chemTabFunction.context.get()), 0);

            ASSERT_FLOAT_EQ(tChemComputedProperty, chemTabComputedProperty) << " The TChem and ChemTab " << testProperty << " should be equal";

            // test the function where temperature is an input
            auto tChemFunctionTemperature = tchem->GetThermodynamicTemperatureMassFractionFunction(testProperty, tChemFields);
            ASSERT_EQ(
                tChemFunctionTemperature.function(tChemFieldsConserved.data(), expectedMassFractions.data(), tChemComputedTemperature, &tChemComputedProperty, tChemFunctionTemperature.context.get()),
                0);

            auto chemTabFunctionTemperature = chemTab->GetThermodynamicTemperatureFunction(testProperty, chemTabFields);
            ASSERT_EQ(chemTabFunctionTemperature.function(chemTabFieldsConserved.data(), chemTabComputedTemperature, &chemTabComputedProperty, chemTabFunctionTemperature.context.get()), 0);

            ASSERT_FLOAT_EQ(tChemComputedProperty, chemTabComputedProperty) << " The TChem and ChemTab " << testProperty << " should be equal when computed with temperature";
        }
    }
}

/*******************************************************************************************************
 * Tests for getting the Progress Variables
 */
TEST_P(ChemTabTestFixture, ShouldComputeCorrectProgressVariables) {
    ONLY_WITH_TENSORFLOW_CHECK;

    for (const auto& testTarget : testTargets) {
        // arrange
        ablate::eos::ChemTab chemTabModel(GetParam().modelPath);
        auto expectedProgressVariables = testTarget["output_cpvs"].as<std::vector<double>>();
        auto inputMassFractions = testTarget["input_mass_fractions"].as<std::vector<double>>();

        // act
        // Size up the results based upon expected
        std::vector<PetscReal> actual(expectedProgressVariables.size());
        chemTabModel.ComputeProgressVariables(inputMassFractions, actual);

        // assert
        for (std::size_t r = 0; r < actual.size(); r++) {
            assert_float_close(expectedProgressVariables[r], actual[r]) << "The value for input set [" << r << "] is incorrect for model " << testTarget["testName"].as<std::string>();
        }
    }
}

TEST_P(ChemTabTestFixture, ShouldComputeFieldFromProgressVariable) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    auto chemTab = std::make_shared<ablate::eos::ChemTab>(GetParam().modelPath);

    // build a new reference eos
    auto metadata = YAML::LoadFile(std::filesystem::path(GetParam().modelPath) / "metadata.yaml");
    auto tchem = std::make_shared<ablate::eos::TChem>(std::filesystem::path(GetParam().modelPath) / metadata["mechanism"].as<std::string>());

    // Get the list of initializers
    std::map<std::string, std::map<std::string, double>> initializers;
    for (const auto& node : metadata["initializers"]) {
        initializers[node.first.as<std::string>()] = node.second["mass_fractions"].as<std::map<std::string, double>>();
    }

    ASSERT_TRUE(initializers.size() > 1) << "All ChemTab models should have at least two initializers";

    // March over each initializer
    for (const auto& [label, speciesMap] : initializers) {
        // compute yi scratch for tchem
        std::vector<PetscReal> yi(tchem->GetSpeciesVariables().size(), 0.0);
        for (const auto& [species, value] : speciesMap) {
            auto loc = std::find(tchem->GetSpeciesVariables().begin(), tchem->GetSpeciesVariables().end(), species);
            if (loc == tchem->GetSpeciesVariables().end()) {
                throw std::invalid_argument("Unable to locate species " + species);
            }
            yi[std::distance(tchem->GetSpeciesVariables().begin(), loc)] = value;
        }

        std::vector<PetscReal> expectedProgressVariable(chemTab->GetProgressVariables().size());
        chemTab->ComputeProgressVariables(yi, expectedProgressVariable);

        // Get the actual progress variable values for this initializers
        std::vector<PetscReal> actualProgressVariable;
        chemTab->GetInitializerProgressVariables(label, actualProgressVariable);

        // assert
        for (std::size_t c = 0; c < actualProgressVariable.size(); c++) {
            ASSERT_EQ(actualProgressVariable[c], expectedProgressVariable[c]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabTests, ChemTabTestFixture,
                         testing::Values((ChemTabTestParameters){.modelPath = "inputs/eos/chemTabTestModel_1", .testTargetFile = "inputs/eos/chemTabTestModel_1/testTargets.yaml"}));

/*********************************************************************************************************
 * Test for when tensorflow is not available
 */
TEST(ChemTabTests, ShouldReportTensorFlowLibraryMissing) {
    ONLY_WITHOUT_TENSORFLOW_CHECK;
    ASSERT_ANY_THROW(ablate::eos::ChemTab("inputs/eos/chemTabTestModel_1"));
}

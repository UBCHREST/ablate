#include <yaml-cpp/yaml.h>
#include "chemistry/chemTabModel.hpp"
#include "gtest/gtest.h"
#include "localPath.hpp"
#include "mockFactory.hpp"
#include "parameters/mapParameters.hpp"

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
TEST(ChemTabModelTests, ShouldCreateFromRegistar) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    std::shared_ptr<cppParserTesting::MockFactory> mockFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedClassType = "ablate::chemistry::ChemTabModel";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));

    auto mockSubFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedSubClassType = "";
    EXPECT_CALL(*mockSubFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedSubClassType));
    EXPECT_CALL(*mockSubFactory, Get(cppParser::ArgumentIdentifier<std::string>{})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("inputs/chemistry/chemTabTestModel_1"));
    EXPECT_CALL(*mockFactory, GetFactory("path")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(mockSubFactory));

    // act
    auto createMethod = Creator<ablate::chemistry::ChemistryModel>::GetCreateMethod(mockFactory->GetClassType());
    auto instance = createMethod(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the ablate::chemistry::ChemTabModel";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::chemistry::ChemTabModel>(instance) != nullptr) << " should be an instance of ablate::chemistry::ChemTabModel";
}

/*******************************************************************************************************
 * Tests for expected input/outputs
 */
struct ChemTabModelTestParameters {
    std::string modelPath;
    std::string testTargetFile;
};
class ChemTabModelTestFixture : public testing::TestWithParam<ChemTabModelTestParameters> {
   protected:
    YAML::Node testTargets;

    void SetUp() override {
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
        ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);

        // act
        auto actualSpecies = chemTabModel.GetSpecies();
        auto actualProgressVariables = chemTabModel.GetProgressVariables();

        // assert
        EXPECT_EQ(testTarget["species_names"].as<std::vector<std::string>>(), actualSpecies) << "should compute correct species name for model " << testTarget["testName"].as<std::string>();
        EXPECT_EQ(testTarget["cpv_names"].as<std::vector<std::string>>(), actualProgressVariables) << "should compute correct cpv names for model " << testTarget["testName"].as<std::string>();
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
        ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);
        auto chemTabModelComputeMassFractionsFunction = chemTabModel.GetComputeMassFractionsFunction();
        auto ctx = chemTabModel.GetContext();
        auto expectedMassFractions = testTarget["output_mass_fractions"].as<std::vector<double>>();
        auto inputProgressVariables = testTarget["input_cpvs"].as<std::vector<double>>();

        // act
        std::vector<PetscReal> actual(expectedMassFractions.size());
        chemTabModelComputeMassFractionsFunction(inputProgressVariables.data(), inputProgressVariables.size(), actual.data(), actual.size(), ctx);

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
        ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);
        auto chemTabModelComputeSourceFunction = chemTabModel.GetComputeSourceFunction();
        auto ctx = chemTabModel.GetContext();
        auto inputProgressVariables = testTarget["input_cpvs"].as<std::vector<double>>();
        auto expectedSourceEnergy = testTarget["output_source_energy"].as<double>();
        auto expectedSource = testTarget["output_source_terms"].as<std::vector<double>>();

        // act
        // Size up the results based upon expected
        std::vector<PetscReal> actual(expectedSource.size());
        PetscReal actualSourceEnergy;
        chemTabModelComputeSourceFunction(inputProgressVariables.data(), inputProgressVariables.size(), &actualSourceEnergy, actual.data(), actual.size(), ctx);

        assert_float_close(expectedSourceEnergy, actualSourceEnergy) << "The sourceEnergy is incorrect for model " << testTarget["testName"].as<std::string>();

        for (std::size_t r = 0; r < actual.size(); r++) {
            assert_float_close(expectedSource[r], actual[r]) << " the percent difference of (" << expectedSource[r] << ", " << actual[r] << ") should be less than 5.0E-6 for index [" << r
                                                             << "] for model " << testTarget["testName"].as<std::string>();
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
        ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);
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

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelTestFixture,
                         testing::Values((ChemTabModelTestParameters){.modelPath = "inputs/chemistry/chemTabTestModel_1", .testTargetFile = "inputs/chemistry/chemTabTestModel_1/testTargets.yaml"}));

/*********************************************************************************************************
 * Test for when tensorflow is not available
 */
TEST(ChemTabModelTests, ShouldReportTensorFlowLibraryMissing) {
    ONLY_WITHOUT_TENSORFLOW_CHECK;
    ASSERT_ANY_THROW(ablate::chemistry::ChemTabModel("inputs/chemistry/chemTabTestModel_1"));
}

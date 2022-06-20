#include "chemistry/chemTabModel.hpp"
#include "gtest/gtest.h"
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

// verified working template (now for auto-generation)
#include "test_targets.h"

/*******************************************************************************************************
 * This test ensure that the chemTabModel can be created using the input file
 */
TEST(ChemTabModelTests, ShouldCreateFromRegistar) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    std::shared_ptr<cppParserTesting::MockFactory> mockFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedClassType = "ablate::chemistry::ChemTabModel";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::filesystem::path>{.inputName = "path"}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return("inputs/chemistry/chemTabTestModel_1"));

    // act
    auto instance = ResolveAndCreate<ablate::chemistry::ChemistryModel>(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the ablate::chemistry::ChemTabModel";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::chemistry::ChemTabModel>(instance) != nullptr) << " should be an instance of ablate::chemistry::ChemTabModel";
}

/*******************************************************************************************************
 * Tests for getting the species and progress variables
 */
struct ChemTabModelGetSpeciesAndProgressVariableTestParameters {
    std::string modelPath;
    std::vector<std::string> expectedSpecies;
    std::vector<std::string> expectedProgressVariables;
};
class ChemTabModelGetSpeciesAndProgressVariableTestFixture : public testing::TestWithParam<ChemTabModelGetSpeciesAndProgressVariableTestParameters> {};

TEST_P(ChemTabModelGetSpeciesAndProgressVariableTestFixture, ShouldReturnCorrectSpeciesAndVariables) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);

    // act
    auto actualSpecies = chemTabModel.GetSpecies();
    auto actualProgressVariables = chemTabModel.GetProgressVariables();

    // assert
    EXPECT_EQ(GetParam().expectedSpecies, actualSpecies);
    EXPECT_EQ(GetParam().expectedProgressVariables, actualProgressVariables);
}

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelGetSpeciesAndProgressVariableTestFixture,
                         testing::Values((ChemTabModelGetSpeciesAndProgressVariableTestParameters){
                             .modelPath = "inputs/chemistry/chemTabTestModel_1", .expectedSpecies = {SPECIES_NAMES}, .expectedProgressVariables = {CPV_NAMES}}));

/*******************************************************************************************************
 * Tests for getting the Compute Mass Fractions Functions
 */
struct ChemTabModelComputeMassFractionsFunctionParameters {
    std::string modelPath;
    std::vector<PetscReal> inputProgressVariables;
    std::vector<PetscReal> expectedMassFractions;
};
class ChemTabModelComputeMassFractionsFunctionFixture : public testing::TestWithParam<ChemTabModelComputeMassFractionsFunctionParameters> {};

TEST_P(ChemTabModelComputeMassFractionsFunctionFixture, ShouldComputeCorrectMassFractions) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);
    auto chemTabModelComputeMassFractionsFunction = chemTabModel.GetComputeMassFractionsFunction();
    auto ctx = chemTabModel.GetContext();

    // act
    std::vector<PetscReal> actual(GetParam().expectedMassFractions.size());
    chemTabModelComputeMassFractionsFunction(GetParam().inputProgressVariables.data(), GetParam().inputProgressVariables.size(), actual.data(), actual.size(), ctx);

    // assert
    for (std::size_t r = 0; r < actual.size(); r++) {
        EXPECT_FLOAT_EQ(GetParam().expectedMassFractions[r], actual[r]) << "The value for [" << r << "] is incorrect";
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelComputeMassFractionsFunctionFixture,
                         testing::Values((ChemTabModelComputeMassFractionsFunctionParameters){
                             .modelPath = "inputs/chemistry/chemTabTestModel_1", .inputProgressVariables = {INPUT_CPVS}, .expectedMassFractions = {OUTPUT_MASS_FRACTIONS}}));

/*******************************************************************************************************
 * Tests for getting the Source and Source Energy Predictions
 */
struct ChemTabModelComputeSourceFunctionParameters {
    std::string modelPath;
    std::vector<PetscReal> inputProgressVariables;
    std::vector<PetscReal> expectedSource;
    PetscReal expectedSourceEnergy;
};
class ChemTabModelComputeSourceFunctionFixture : public testing::TestWithParam<ChemTabModelComputeSourceFunctionParameters> {};

TEST_P(ChemTabModelComputeSourceFunctionFixture, ShouldComputeCorrectSource) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);
    auto chemTabModelComputeSourceFunction = chemTabModel.GetComputeSourceFunction();
    auto ctx = chemTabModel.GetContext();

    // act
    // Size up the results based upon expected
    std::vector<PetscReal> actual(GetParam().expectedSource.size());
    PetscReal actualSourceEnergy;
    chemTabModelComputeSourceFunction(GetParam().inputProgressVariables.data(), GetParam().inputProgressVariables.size(), &actualSourceEnergy, actual.data(), actual.size(), ctx);

    // assert
    EXPECT_FLOAT_EQ(GetParam().expectedSourceEnergy, actualSourceEnergy) << "The sourceEnergy is incorrect";
    for (std::size_t r = 0; r < actual.size(); r++) {
        auto percentDifference = PetscAbs((GetParam().expectedSource[r] - actual[r]) / (0.5 * (GetParam().expectedSource[r] + actual[r])));
        ASSERT_LT(percentDifference, 5.0E-6) << " the percent difference of (" << GetParam().expectedSource[r] << ", " << actual[r] << ") should be less than 5.0E-6 for index [" << r << "] ";
        // EXPECT_FLOAT_EQ(GetParam().expectedSource[r], actual[r]) << "The value for index [" << r << "] is incorrect";
    }
}

INSTANTIATE_TEST_SUITE_P(
    ChemTabModelTests, ChemTabModelComputeSourceFunctionFixture,
    testing::Values((ChemTabModelComputeSourceFunctionParameters){
        .modelPath = "inputs/chemistry/chemTabTestModel_1", .inputProgressVariables = {INPUT_CPVS}, .expectedSource = {OUTPUT_SOURCE_TERMS}, .expectedSourceEnergy = OUTPUT_SOURCE_ENERGY}));

/*******************************************************************************************************
 * Tests for getting the Progress Variables
 */
struct ChemTabModelComputeProgressVariablesParameters {
    std::string modelPath;
    std::vector<PetscReal> inputMassFractions;
    std::vector<PetscReal> expectedProgressVariables;
};
class ChemTabModelComputeProgressVariablesFixture : public testing::TestWithParam<ChemTabModelComputeProgressVariablesParameters> {};

TEST_P(ChemTabModelComputeProgressVariablesFixture, ShouldComputeCorrectProgressVariables) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);

    // act
    // Size up the results based upon expected
    std::vector<PetscReal> actual(GetParam().expectedProgressVariables.size());
    chemTabModel.ComputeProgressVariables(GetParam().inputMassFractions.data(), GetParam().inputMassFractions.size(), actual.data(), actual.size());

    // assert
    for (std::size_t r = 0; r < actual.size(); r++) {
        EXPECT_FLOAT_EQ(GetParam().expectedProgressVariables[r], actual[r]) << "The value for input set [" << r << "] is incorrect";
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelComputeProgressVariablesFixture,
                         testing::Values((ChemTabModelComputeProgressVariablesParameters){
                             .modelPath = "inputs/chemistry/chemTabTestModel_1", .inputMassFractions = {INPUT_MASS_FRACTIONS}, .expectedProgressVariables = {OUTPUT_CPVS}}));

/*********************************************************************************************************
 * Test for when tensorflow is not available
 */
TEST(ChemTabModelTests, ShouldReportTensorFlowLibraryMissing) {
    ONLY_WITHOUT_TENSORFLOW_CHECK;
    ASSERT_ANY_THROW(ablate::chemistry::ChemTabModel("inputs/chemistry/chemTabTestModel_1"));
}

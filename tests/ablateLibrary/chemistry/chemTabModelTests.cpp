#include "chemistry/chemTabModel.hpp"
#include "gtest/gtest.h"
#include "mockFactory.hpp"
#include "parameters/mapParameters.hpp"

#ifndef WITH_TENSORFLOW
#define ONLY_WITH_TENSORFLOW_CHECK GTEST_SKIP_("Test is only applicable with built with TensorFlow")
#define ONLY_WITHOUT_TENSORFLOW_CHECK \
    {}
#else
#define ONLY_WITH_TENSORFLOW_CHECK \
    {}
#define ONLY_WITHOUT_TENSORFLOW_CHECK GTEST_SKIP_("Test is only applicable with built without TensorFlow")
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
                             .modelPath = "inputs/chemistry/chemTabTestModel_1", .expectedSpecies = {"N2", "ect..."}, .expectedProgressVariables = {"Zmix", "ect..."}}));

/*******************************************************************************************************
 * Tests for getting the Compute Mass Fractions Functions
 */
struct ChemTabModelComputeComputeMassFractionsFunctionParameters {
    std::string modelPath;
    std::vector<std::vector<PetscReal>> inputProgressVariables;
    std::vector<std::vector<PetscReal>> expectedMassFractions;
};
class ChemTabModelComputeComputeMassFractionsFunctionFixture : public testing::TestWithParam<ChemTabModelComputeComputeMassFractionsFunctionParameters> {};

TEST_P(ChemTabModelComputeComputeMassFractionsFunctionFixture, ShouldComputeCorrectMassFractions) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);
    auto chemTabModelComputeMassFractionsFunction = chemTabModel.GetComputeMassFractionsFunction();
    auto ctx = chemTabModel.GetContext();

    // act
    for (std::size_t i = 0; i < GetParam().inputProgressVariables.size(); i++) {
        // Size up the results based upon expected
        std::vector<PetscReal> actual(GetParam().expectedMassFractions[i].size());

        chemTabModelComputeMassFractionsFunction(GetParam().inputProgressVariables[i].data(), actual.data(), ctx);

        // assert
        for (std::size_t r = 0; r < actual.size(); r++) {
            EXPECT_DOUBLE_EQ(GetParam().expectedMassFractions[i][r], actual[r]) << "The value for input set [" << i << "] index [" << r << "] is incorrect";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelComputeComputeMassFractionsFunctionFixture,
                         testing::Values((ChemTabModelComputeComputeMassFractionsFunctionParameters){.modelPath = "inputs/chemistry/chemTabTestModel_1",
                                                                                                     .inputProgressVariables = {/*set 0*/ {.1, .2, .3}, /*set 1*/ {.1, .2, .3}},
                                                                                                     .expectedMassFractions = {/*set 0*/ {.5, .6, .7, .8}, /*set 1*/ {.6, .7, .8, .9}}}));

/*******************************************************************************************************
 * Tests for getting the Compute Source Function Functions
 */
struct ChemTabModelComputeComputeSourceFunctionParameters {
    std::string modelPath;
    std::vector<std::vector<PetscReal>> inputProgressVariables;
    std::vector<std::vector<PetscReal>> expectedSource;
    std::vector<PetscReal> expectedSourceEnergy;
};
class ChemTabModelComputeComputeSourceFunctionFixture : public testing::TestWithParam<ChemTabModelComputeComputeSourceFunctionParameters> {};

TEST_P(ChemTabModelComputeComputeSourceFunctionFixture, ShouldComputeCorrectMassFractions) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);
    auto chemTabModelComputeSourceFunction = chemTabModel.GetComputeSourceFunction();
    auto ctx = chemTabModel.GetContext();

    // act
    for (std::size_t i = 0; i < GetParam().inputProgressVariables.size(); i++) {
        // Size up the results based upon expected
        std::vector<PetscReal> actual(GetParam().expectedSource[i].size());
        PetscReal actualSourceEnergy;
        chemTabModelComputeSourceFunction(GetParam().inputProgressVariables[i].data(), actualSourceEnergy, actual.data(), ctx);

        // assert
        EXPECT_DOUBLE_EQ(GetParam().expectedSourceEnergy[i], actualSourceEnergy) << "The sourceEnergy for input set [" << i << "] is incorrect";
        for (std::size_t r = 0; r < actual.size(); r++) {
            EXPECT_DOUBLE_EQ(GetParam().expectedSource[i][r], actual[r]) << "The value for input set [" << i << "] index [" << r << "] is incorrect";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelComputeComputeSourceFunctionFixture,
                         testing::Values((ChemTabModelComputeComputeSourceFunctionParameters){.modelPath = "inputs/chemistry/chemTabTestModel_1",
                                                                                              .inputProgressVariables = {/*set 0*/ {.1, .2, .3}, /*set 1*/ {.1, .2, .3}},
                                                                                              .expectedSource = {/*set 0*/ {.5, .6, .7, .8}, /*set 1*/ {.6, .7, .8, .9}},
                                                                                              .expectedSourceEnergy = {200, 300}}));

/*******************************************************************************************************
 * Tests for getting the Compute Source Function Functions
 */
struct ChemTabModelComputeProgressVariablesParameters {
    std::string modelPath;
    std::vector<std::vector<PetscReal>> inputMassFractions;
    std::vector<std::vector<PetscReal>> expectedProgressVariables;
};
class ChemTabModelComputeProgressVariablesFixture : public testing::TestWithParam<ChemTabModelComputeProgressVariablesParameters> {};

TEST_P(ChemTabModelComputeProgressVariablesFixture, ShouldComputeCorrectMassFractions) {
    ONLY_WITH_TENSORFLOW_CHECK;

    // arrange
    ablate::chemistry::ChemTabModel chemTabModel(GetParam().modelPath);

    // act
    for (std::size_t i = 0; i < GetParam().inputMassFractions.size(); i++) {
        // Size up the results based upon expected
        std::vector<PetscReal> actual(GetParam().expectedProgressVariables[i].size());
        PetscReal actualSourceEnergy;
        chemTabModel.ComputeProgressVariables(GetParam().inputMassFractions[i].data(), actual.data());

        // assert
        for (std::size_t r = 0; r < actual.size(); r++) {
            EXPECT_DOUBLE_EQ(GetParam().expectedProgressVariables[i][r], actual[r]) << "The value for input set [" << i << "] index [" << r << "] is incorrect";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelComputeProgressVariablesFixture,
                         testing::Values((ChemTabModelComputeProgressVariablesParameters){.modelPath = "inputs/chemistry/chemTabTestModel_1",
                                                                                          .inputMassFractions = {/*set 0*/ {.1, .2, .3}, /*set 1*/ {.1, .2, .3}},
                                                                                          .expectedProgressVariables = {/*set 0*/ {.5, .6, .7, .8}, /*set 1*/ {.6, .7, .8, .9}}}));

/*********************************************************************************************************
 * Test for when tensorflow is not available
 */
TEST(ChemTabModelTests, ShouldReportTensorFlowLibraryMissing) {
    ONLY_WITHOUT_TENSORFLOW_CHECK;
    ASSERT_ANY_THROW(ablate::chemistry::ChemTabModel("inputs/chemistry/chemTabTestModel_1"));
}
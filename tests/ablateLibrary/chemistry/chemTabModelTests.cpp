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
                             .modelPath = "inputs/chemistry/chemTabTestModel_1",
                             .expectedSpecies = {"YiCH4",   "YiH",     "YiO",    "YiO2",   "YiOH",    "YiH2O",  "YiHO2",   "YiH2O2",   "YiC",      "YiCH",   "YiCH2",  "YiCH2(S)", "YiCH3",  "YiH2",
                                                 "YiCO",    "YiCO2",   "YiHCO",  "YiCH2O", "YiCH2OH", "YiCH3O", "YiCH3OH", "YiC2H",    "YiC2H2",   "YiC2H3", "YiC2H4", "YiC2H5",   "YiC2H6", "YiHCCO",
                                                 "YiCH2CO", "YiHCCOH", "YiN",    "YiNH",   "YiNH2",   "YiNH3",  "YiNNH",   "YiNO",     "YiNO2",    "YiN2O",  "YiHNO",  "YiCN",     "YiHCN",  "YiH2CN",
                                                 "YiHCNN",  "YiHCNO",  "YiHOCN", "YiHNCO", "YiNCO",   "YiC3H7", "YiC3H8",  "YiCH2CHO", "YiCH3CHO", "YiN2",   "YiAR"},
                             .expectedProgressVariables = {"zmix", "CPV_0", "CPV_1", "CPV_2", "CPV_3"}}));

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
                             .modelPath = "inputs/chemistry/chemTabTestModel_1",
                             .inputProgressVariables = {0, .1, .2, .2, .1},
                             .expectedMassFractions = {0.024346507649162213,  0.03070802663635961,  -0.014177157863655775, -0.012206614867194465,  -0.009586717144079401,  -0.005967521119935231,
                                                       -0.010875227678139601, -0.03214015967582351, 0.028280729183712428,  0.01591420682377494,    0.01276966432475931,    0.016103929505397983,
                                                       0.02193572707541392,   0.04443951471120011,  0.0002556383726955288, -0.0010179648987351305, -0.00786316213077087,   0.002096940503499309,
                                                       0.010562042434247647,  0.016693196037656108, 0.025797843813450662,  0.02036901883527415,    0.019451247923288117,   0.02054569299637939,
                                                       0.020019336518559503,  0.01835249376645227,  0.021750708609175502,  0.0018026983778071494,  0.010790575842471367,   0.016842113379018253,
                                                       0.033468867920825464,  0.018857471749292653, 0.044804958092783155,  0.13419273882405358,    -0.0037252238295265167, -0.06779158472561372,
                                                       -0.05588082676187539,  -0.048957981486385,   -0.0984548580421235,   0.017779474312028466,   -0.025172586623901604,  -0.006983341400292928,
                                                       -0.08826244329397101,  0.029648167069252704, 0.038806576447438,     -0.0177198587682406,    -0.09435747073550743,   0.02751871209625231,
                                                       0.020135538237014753,  0.04394390344680865,  -0.04396782613129059,  0.051775990887332514,   0.11555003804381765}}));

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

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelComputeSourceFunctionFixture,
                         testing::Values((ChemTabModelComputeSourceFunctionParameters){.modelPath = "inputs/chemistry/chemTabTestModel_1",
                                                                                       .inputProgressVariables = {1., .1, .2, .2, .1},
                                                                                       .expectedSource = {39.918297, -32.271481, -26.513624, 36.252228},
                                                                                       .expectedSourceEnergy = -3.1341994e+12}));

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
                             .modelPath = "inputs/chemistry/chemTabTestModel_1",
                             .inputMassFractions = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.,  2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                                                    2.8, 2.9, 3.,  3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.,  4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.,  5.1, 5.2, 5.3},
                             .expectedProgressVariables = {0, 0.3306785324000005, -5.4590796122, 1.3819009245500002, 7.1005789664000005}}));

/*********************************************************************************************************
 * Test for when tensorflow is not available
 */
TEST(ChemTabModelTests, ShouldReportTensorFlowLibraryMissing) {
    ONLY_WITHOUT_TENSORFLOW_CHECK;
    ASSERT_ANY_THROW(ablate::chemistry::ChemTabModel("inputs/chemistry/chemTabTestModel_1"));
}

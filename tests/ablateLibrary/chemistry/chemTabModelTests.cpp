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
                             .modelPath = "inputs/chemistry/chemTabTestModel_1",
                             .expectedSpecies = {"YiCH4",   "YiH",     "YiO",    "YiO2",   "YiOH",    "YiH2O",  "YiHO2",   "YiH2O2",   "YiC",      "YiCH",   "YiCH2",  "YiCH2(S)", "YiCH3",  "YiH2",
                                                 "YiCO",    "YiCO2",   "YiHCO",  "YiCH2O", "YiCH2OH", "YiCH3O", "YiCH3OH", "YiC2H",    "YiC2H2",   "YiC2H3", "YiC2H4", "YiC2H5",   "YiC2H6", "YiHCCO",
                                                 "YiCH2CO", "YiHCCOH", "YiN",    "YiNH",   "YiNH2",   "YiNH3",  "YiNNH",   "YiNO",     "YiNO2",    "YiN2O",  "YiHNO",  "YiCN",     "YiHCN",  "YiH2CN",
                                                 "YiHCNN",  "YiHCNO",  "YiHOCN", "YiHNCO", "YiNCO",   "YiC3H7", "YiC3H8",  "YiCH2CHO", "YiCH3CHO", "YiN2",   "YiAR"},
                             .expectedProgressVariables = {"zmix", "CPV_0", "CPV_1"}}));

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
                             .inputProgressVariables = {0,.1, .2},
                             .expectedMassFractions = {
                                 -0.00140725565130274,    -0.00045837323449758003, -0.00035455781428995987, -0.0013910013371491,    -0.0002487882385107601, -2.04598882374001e-06,
                                 -0.00038961941524638996, -2.7069681346739975e-05, -0.0005482163425314899,  0.00105295943562432,    -0.0035183326637905103, -0.0005537085981488,
                                 0.002645006946767,       0.00012232255899329998,  3.0051809915890072e-05,  -6.642647683107998e-05, 0.00087529355466593,    -0.00171778095303233,
                                 -0.0012801298701463797,  -4.033892815447998e-05,  0.00023066407677046002,  -0.00136504469385059,   0.0015772327251534102,  -0.00078278179181311,
                                 -0.00019032477892702,    -0.00023789371908623,    -0.00534847132631586,    -0.00107020781603896,   0.0007697984124302801,  0.0026045736831105497,
                                 -0.0011341133827743601,  -0.00075079009863218,    0.0005390351054686101,   -0.00022974845536523,   0.00010588862332675,    -0.00019050340742513,
                                 -0.00023533306406412,    -0.00028523556122276,    0.00014006160875666002,  -0.00032022301953658,   -0.0001956971517322,    -0.00048963887131575,
                                 -5.431705411270023e-06,  0.00055470396011405,     0.0008781291099871601,   -4.286525508170001e-05, 8.478539365690017e-06,  -0.0022354488831264796,
                                 0.00192641706847605,     0.00074413982307823,     0.00010748827212834991,  -7.853775655976e-05,    5.923708549286434e-05}}));

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
        EXPECT_FLOAT_EQ(GetParam().expectedSource[r], actual[r]) << "The value for index [" << r << "] is incorrect";
    }
}

INSTANTIATE_TEST_SUITE_P(ChemTabModelTests, ChemTabModelComputeSourceFunctionFixture,
                         testing::Values((ChemTabModelComputeSourceFunctionParameters){
                             .modelPath = "inputs/chemistry/chemTabTestModel_1", .inputProgressVariables = {1.0, .1, .2}, .expectedSource = {0.0, 0, 0}, .expectedSourceEnergy = 233370.5}));

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
                             //.expectedProgressVariables = {-17.845192, 47.81491243, -58.08515966}}));
                             .expectedProgressVariables = {0, 47.81491243, -58.08515966}}));

/*********************************************************************************************************
 * Test for when tensorflow is not available
 */
TEST(ChemTabModelTests, ShouldReportTensorFlowLibraryMissing) {
    ONLY_WITHOUT_TENSORFLOW_CHECK;
    ASSERT_ANY_THROW(ablate::chemistry::ChemTabModel("inputs/chemistry/chemTabTestModel_1"));
}

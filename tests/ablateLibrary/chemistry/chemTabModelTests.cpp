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
                             .expectedProgressVariables = {"zmix", "CPV_0", "CPV_1", "CPV_2", "CPV_3", "CPV_4", "CPV_5", "CPV_6", "CPV_7", "CPV_8", "CPV_9"}}));

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
			     .inputProgressVariables = {0.1,0.2,0.30000000000000004,0.4,0.5,0.6,0.7000000000000001,0.8,0.9,1.0},
			     .expectedMassFractions = {
                 0.08729344067400002,-0.057736426000000035,-0.0507710609,-0.05038019370000001,0.022414269831999983,0.034474443529999996,-0.05198052570000003,0.48068658230000005,0.14977136371000005,0.17721069489999994,0.1803145761,-0.021572660600000015,0.2425669009,0.022014798400000077,0.04403852501000001,0.0072819911999999765,0.30069876151899994,-0.037637864500000034,0.12942771480000004,0.2588041834,0.043532963799999935,0.1517175009,0.11970174330000004,0.12321197460000008,0.13906382850000001,0.051165023000000004,0.13919231099999999,0.025434686799999967,0.07436949060000003,-0.1651681889999999,0.03707771293999995,-0.48263475930000005,0.46724466620000005,0.311447044,-0.38969315260000004,-0.11576112499999994,-0.156160855,0.05602832049999999,0.31638452949999996,-0.11571994840000004,-0.073598585,0.0831601457,0.5815045736,0.6290163077,0.0411177076,-0.10481887839999998,-0.04744206825000001,0.6253895152,0.4906405896,0.06096917730000001,-0.08332413899999994,-0.1831054178,0.018453095999999974
                 }}));

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
		testing::Values((ChemTabModelComputeSourceFunctionParameters){
			.modelPath = "inputs/chemistry/chemTabTestModel_1", .inputProgressVariables = {0.1,0.2,0.30000000000000004,0.4,0.5,0.6,0.7000000000000001,0.8,0.9,1.0}, .expectedSource = {
            -761.25995,959.7151,-2545.9788,1402.5092,1277.0408,-652.83374,-508.75754,-782.3674,504.25546,1156.204
            }, .expectedSourceEnergy = 249049450000.0}));

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
			.expectedProgressVariables = {13.219376968116123,10.046072769941105,17.27133197341372,13.19597183205257,9.60662474951432,16.274740548561393,14.245004801708909,12.53670218761842,10.573577940048557,15.96326180505703}}));

			/*********************************************************************************************************
			 * Test for when tensorflow is not available
			 */
			TEST(ChemTabModelTests, ShouldReportTensorFlowLibraryMissing) {
				ONLY_WITHOUT_TENSORFLOW_CHECK;
				ASSERT_ANY_THROW(ablate::chemistry::ChemTabModel("inputs/chemistry/chemTabTestModel_1"));
			}

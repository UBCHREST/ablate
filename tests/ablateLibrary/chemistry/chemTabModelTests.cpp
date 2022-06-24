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

// verified working template (now for auto-generation)
#define CPV_NAMES "zmix", "CPV_0", "CPV_1", "CPV_2", "CPV_3", "CPV_4", "CPV_5", "CPV_6", "CPV_7"
#define SPECIES_NAMES                                                                                                                                                                            \
    "YiCH4", "YiH", "YiO", "YiO2", "YiOH", "YiH2O", "YiHO2", "YiH2O2", "YiC", "YiCH", "YiCH2", "YiCH2(S)", "YiCH3", "YiH2", "YiCO", "YiCO2", "YiHCO", "YiCH2O", "YiCH2OH", "YiCH3O", "YiCH3OH",  \
        "YiC2H", "YiC2H2", "YiC2H3", "YiC2H4", "YiC2H5", "YiC2H6", "YiHCCO", "YiCH2CO", "YiHCCOH", "YiN", "YiNH", "YiNH2", "YiNH3", "YiNNH", "YiNO", "YiNO2", "YiN2O", "YiHNO", "YiCN", "YiHCN", \
        "YiH2CN", "YiHCNN", "YiHCNO", "YiHOCN", "YiHNCO", "YiNCO", "YiC3H7", "YiC3H8", "YiCH2CHO", "YiCH3CHO", "YiN2", "YiAR"

#define INPUT_CPVS 0.0, 0.1, 0.2285714285714286, 0.3571428571428572, 0.48571428571428577, 0.6142857142857143, 0.7428571428571429, 0.8714285714285716, 1.0
#define OUTPUT_MASS_FRACTIONS                                                                                                                                                                          \
    0.29461223846183565, 0.14894779675801886, 0.05505332862768736, 0.16690875020705553, 0.0264104776979555, 0.1710982028611339, 0.037449645715917995, 0.04446187298777576, 0.11071041384132632,        \
        0.41266943630659747, 0.17911240010044063, 0.24892078879420831, 0.4373546105760766, 0.09014362051977609, 0.033448514401321384, 0.02146226110926706, 0.005069577863025714, 0.029002364584854463, \
        0.10772134218901001, 0.15988099292317803, 0.4017263714858558, 0.09331639237926637, 0.0639114513422509, 0.08432514551921173, 0.09360514002128369, 0.04890451142284812, 0.18421555436800932,     \
        0.02101629329071502, 0.2638469221168997, 0.24636483053427682, 0.047392473023381176, 0.37678893352199083, 0.4695726894873439, 0.2857165414979913, 0.47044485017187876, 0.31585504775623124,     \
        0.11185135824721884, 0.5436719198240042, 0.038373787021688456, 0.17454631505861407, 0.25583303581959216, 0.2532242043566313, 0.0015759721542994765, 0.5137773903766832, 0.30877460412133256,   \
        0.3391418535447704, 0.36108736322417245, 0.0317790310230259, 0.12865598651316973, 0.020601319749237065, 0.3368246344503261, 0.11514581623927923, 0.3482837663576783

#define INPUT_MASS_FRACTIONS                                                                                                                                                                          \
    0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6, 0.7000000000000001, 0.8, 0.9, 1.0, 1.1, 1.2000000000000002, 1.3000000000000003, 1.4000000000000001, 1.5000000000000002, 1.6, 1.7000000000000002,    \
        1.8000000000000003, 1.9000000000000001, 2.0, 2.1, 2.2, 2.3000000000000003, 2.4000000000000004, 2.5000000000000004, 2.6, 2.7, 2.8000000000000003, 2.9000000000000004, 3.0000000000000004, 3.1, \
        3.2, 3.3000000000000003, 3.4000000000000004, 3.5000000000000004, 3.6, 3.7, 3.8000000000000003, 3.9000000000000004, 4.0, 4.1, 4.2, 4.3, 4.3999999999999995, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, \
        5.2, 5.3
#define OUTPUT_CPVS 0.0, 6.690776786566, 5.9750706723365, 6.4911125694635, 6.602055609830001, 5.6165583358134, 7.6665188310312, 7.673165989480001, 7.644177309551001

#define OUTPUT_SOURCE_TERMS 99.387634, -43.034847, -34.94731, 17.041483, -28.672432, 19.51686, -60.53373, -39.277264
#define OUTPUT_SOURCE_ENERGY 15127823360.0

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

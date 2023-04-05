#include <map>
#include <memory>
#include <set>
#include "gtest/gtest.h"
#include "mathFunctions/formula.hpp"
#include "mathFunctions/functionFactory.hpp"
#include "mockFactory.hpp"
#include "parameters/mapParameters.hpp"
#include "registrar.hpp"

namespace ablateTesting::mathFunctions {

TEST(FormulaTests, ShouldBeCreatedFromRegistar) {
    // arrange
    std::shared_ptr<cppParserTesting::MockFactory> mockFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedClassType = "ablate::mathFunctions::Formula";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::string>{.inputName = "formula"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("x+y+z+t"));

    std::shared_ptr<cppParserTesting::MockFactory> childFactory = std::make_shared<cppParserTesting::MockFactory>();
    EXPECT_CALL(*mockFactory, Contains("nested")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
    EXPECT_CALL(*mockFactory, GetFactory("nested")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(childFactory));
    EXPECT_CALL(*childFactory, GetKeys()).Times(::testing::Exactly(1));

    EXPECT_CALL(*mockFactory, Contains("constants")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto createMethod = Creator<ablate::mathFunctions::MathFunction>::GetCreateMethod(mockFactory->GetClassType());
    auto instance = createMethod(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the ParsedNested";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::Formula>(instance) != nullptr) << " should be an instance of ParsedNested";
}

TEST(FormulaTests, ShouldThrowExceptionInvalidEquation) {
    // arrange/act/assert
    ASSERT_THROW(ablate::mathFunctions::Formula("x+y+z+t+c", {}), std::invalid_argument);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> ToFunctionMap(const std::map<std::string, std::string>& stringMap) {
    std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> map;
    for (const auto& formula : stringMap) {
        map[formula.first] = ablate::mathFunctions::Create(formula.second);
    }
    return map;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct FormulaScalarParameters {
    std::string formula;
    std::map<std::string, std::string> nested;
    std::shared_ptr<ablate::parameters::Parameters> constants = nullptr;
    double expectedResult;
};

class FormulaScalarFixture : public ::testing::TestWithParam<FormulaScalarParameters> {};

TEST_P(FormulaScalarFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::Formula(param.formula, ToFunctionMap(param.nested), param.constants);

    // act/assert
    ASSERT_DOUBLE_EQ(param.expectedResult, function.Eval(1.0, 2.0, 3.0, 4.0));
}

TEST_P(FormulaScalarFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::Formula(param.formula, ToFunctionMap(param.nested), param.constants);

    const double array1[3] = {1.0, 2.0, 3.0};

    // act/assert
    ASSERT_DOUBLE_EQ(param.expectedResult, function.Eval(array1, 3, 4.0));
}

INSTANTIATE_TEST_SUITE_P(FormulaTests, FormulaScalarFixture,
                         testing::Values((FormulaScalarParameters){.formula = "v*x", .nested = {{"v", "2.0"}}, .constants = {}, .expectedResult = 2.0},
                                         (FormulaScalarParameters){.formula = "v*x + z", .nested = {{"v", "3.0*y"}}, .constants = {}, .expectedResult = 9.0},
                                         (FormulaScalarParameters){.formula = "t*vel + test", .nested = {{"vel", "3.0*y"}, {"test", "z"}}, .constants = {}, .expectedResult = 27},
                                         (FormulaScalarParameters){.formula = "t*vel + test + CC/AA",
                                                                   .nested = {{"vel", "3.0*y"}, {"test", "z"}},
                                                                   .constants = ablate::parameters::MapParameters::Create({{"CC", "3"}, {"AA", "1.5"}}),
                                                                   .expectedResult = 29},
                                         (FormulaScalarParameters){
                                             .formula = "t*CC/AA", .nested = {}, .constants = ablate::parameters::MapParameters::Create({{"CC", "3"}, {"AA", "1.5"}}), .expectedResult = 8}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct FormulaTestsVectorParameters {
    std::string formula;
    std::map<std::string, std::string> nested;
    std::shared_ptr<ablate::parameters::Parameters> constants = nullptr;
    std::vector<double> expectedResult;
};

class FormulaTestsVectorFixture : public ::testing::TestWithParam<FormulaTestsVectorParameters> {};

TEST_P(FormulaTestsVectorFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::Formula(param.formula, ToFunctionMap(param.nested), param.constants);
    std::vector<double> result(param.expectedResult.size(), NAN);

    // act
    function.Eval(1.0, 2.0, 3.0, 4.0, result);

    // assert
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

TEST_P(FormulaTestsVectorFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::Formula(param.formula, ToFunctionMap(param.nested), param.constants);
    std::vector<double> result(param.expectedResult.size(), NAN);

    const double array[3] = {1.0, 2.0, 3.0};

    // act
    function.Eval(array, 3, 4, result);

    // assert
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

TEST_P(FormulaTestsVectorFixture, ShouldComputeCorrectAnswerPetscFunction) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::Formula(param.formula, ToFunctionMap(param.nested), param.constants);
    std::vector<double> result(param.expectedResult.size(), NAN);

    const double array[3] = {1.0, 2.0, 3.0};

    auto context = function.GetContext();
    auto functionPointer = function.GetPetscFunction();

    // act
    auto errorCode = functionPointer(3, 4.0, array, result.size(), &result[0], context);

    // assert
    ASSERT_EQ(errorCode, 0);
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}
INSTANTIATE_TEST_SUITE_P(
    FormulaTests, FormulaTestsVectorFixture,
    testing::Values((FormulaTestsVectorParameters){.formula = "v*x", .nested = {{"v", "2.0"}}, .constants = {}, .expectedResult = {2.0}},
                    (FormulaTestsVectorParameters){.formula = "v*x + z", .nested = {{"v", "3.0*y"}}, .constants = {}, .expectedResult = {9.0}},
                    (FormulaTestsVectorParameters){.formula = "t*vel + test", .nested = {{"vel", "3.0*y"}, {"test", "z"}}, .constants = {}, .expectedResult = {27}},
                    (FormulaTestsVectorParameters){.formula = "v*x, v*z + y", .nested = {{"v", "3.0*y"}}, .constants = {}, .expectedResult = {6, 20}},
                    (FormulaTestsVectorParameters){.formula = "0, i*y, t/i", .nested = {{"i", "10.0"}}, .constants = {}, .expectedResult = {0, 20, 0.4}},
                    (FormulaTestsVectorParameters){.formula = "0+CC, i*y+CC, t/i+AA",
                                                   .nested = {{"i", "10.0"}},
                                                   .constants = ablate::parameters::MapParameters::Create({{"CC", "3"}, {"AA", "1.5"}}),
                                                   .expectedResult = {3, 23, 1.9}},
                    (FormulaTestsVectorParameters){
                        .formula = "0+CC, y+CC, t+AA", .nested = {}, .constants = ablate::parameters::MapParameters::Create({{"CC", "3"}, {"AA", "1.5"}}), .expectedResult = {3, 5, 5.5}}));

///////////////////////////////////////////////////////////////////////////////////////////////////
TEST(FormulaTests, ShouldProduceDeterministicPsueduRandomNumber) {
    // arrange
    auto functionA = ablate::mathFunctions::Formula(std::string("x*pRand(0, 100)"));
    auto functionB = ablate::mathFunctions::Formula(std::string("y*pRand(0, 100)"));

    std::vector<double> resultsA;
    std::vector<double> resultsB;

    // act
    for (std::size_t i = 0; i < 25; ++i) {
        resultsA.push_back(functionA.Eval(5, 0, 0, 0));
    }
    for (std::size_t i = 0; i < 25; ++i) {
        resultsB.push_back(functionB.Eval(0, 5, 0, 0));
    }

    std::vector<double> expectedResults{229.32506601160992, 109.47959310623946, 339.43235837034274, 467.34644811336943, 259.70818601137375, 17.28605523242372,  264.85009657052859,
                                        3.8490930307995894, 33.421118631283775, 343.38635620452266, 465.21824748484642, 263.46438887995305, 326.95948109026131, 350.59529724958838,
                                        381.09901999931708, 23.732256693155975, 164.11711308000383, 378.20524306629301, 182.66933544578842, 491.27514316051713, 376.67791760279147,
                                        36.342941410115536, 442.35356440792077, 218.20570275781975, 238.86588255920407};

    // assert
    ASSERT_EQ(resultsA, resultsB);
    for (std::size_t i = 0; i < 25; ++i) {
        ASSERT_NEAR(resultsA[i], expectedResults[i], 1E-6) << " the " << i << " pseduo random number is expected to be predictable";
    }
}

TEST(FormulaTests, ShouldProduceRandomNumber) {
    // arrange
    auto functionA = ablate::mathFunctions::Formula(std::string("x*rand(-1, 1)"));
    auto functionB = ablate::mathFunctions::Formula(std::string("y*rand(-1, 1)"));

    std::set<double> resultsA;
    std::set<double> resultsB;

    // act
    for (std::size_t i = 0; i < 25; ++i) {
        resultsA.insert(functionA.Eval(5, 0, 0, 0));
    }
    for (std::size_t i = 0; i < 25; ++i) {
        resultsB.insert(functionB.Eval(0, 5, 0, 0));
    }

    // assert
    ASSERT_EQ(resultsA.size(), 25) << " duplicate random numbers detected";
    ASSERT_EQ(resultsB.size(), 25) << " duplicate random numbers detected";
    ASSERT_NE(resultsA, resultsB);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct FormulaTestsModulusParameters {
    std::string formula;
    double expected;
};

class FormulaTestsModulusFixture : public ::testing::TestWithParam<FormulaTestsModulusParameters> {};

TEST_P(FormulaTestsModulusFixture, ShouldComputeCorrectAnswer) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::Formula(param.formula, {});

    // act
    auto result = function.Eval(0, 0, 0, 0);

    // assert
    ASSERT_NEAR(param.expected, result, 1E-8) << "The modulus for " << param.formula << " should be correct";
}
INSTANTIATE_TEST_SUITE_P(FormulaTests, FormulaTestsModulusFixture,
                         testing::Values((FormulaTestsModulusParameters){.formula = "5 % 2", .expected = 1}, (FormulaTestsModulusParameters){.formula = "100 % 8", .expected = 4},
                                         (FormulaTestsModulusParameters){.formula = "100.1 % 8.2", .expected = 1.7}, (FormulaTestsModulusParameters){.formula = "3*3 % 1+1", .expected = 1},
                                         (FormulaTestsModulusParameters){.formula = "100 % 27 % 4", .expected = 3}));

}  // namespace ablateTesting::mathFunctions
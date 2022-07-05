#include <map>
#include <memory>
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
                         testing::Values((FormulaScalarParameters){.formula = "v*x", .nested = {{"v", "2.0"}}, .expectedResult = 2.0},
                                         (FormulaScalarParameters){.formula = "v*x + z", .nested = {{"v", "3.0*y"}}, .expectedResult = 9.0},
                                         (FormulaScalarParameters){.formula = "t*vel + test", .nested = {{"vel", "3.0*y"}, {"test", "z"}}, .expectedResult = 27},
                                         (FormulaScalarParameters){.formula = "t*vel + test + CC/AA",
                                                                   .nested = {{"vel", "3.0*y"}, {"test", "z"}},
                                                                   .constants = ablate::parameters::MapParameters::Create({{"CC", "3"}, {"AA", "1.5"}}),
                                                                   .expectedResult = 29},
                                         (FormulaScalarParameters){.formula = "t*CC/AA", .constants = ablate::parameters::MapParameters::Create({{"CC", "3"}, {"AA", "1.5"}}), .expectedResult = 8}));

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
INSTANTIATE_TEST_SUITE_P(FormulaTests, FormulaTestsVectorFixture,
                         testing::Values((FormulaTestsVectorParameters){.formula = "v*x", .nested = {{"v", "2.0"}}, .expectedResult = {2.0}},
                                         (FormulaTestsVectorParameters){.formula = "v*x + z", .nested = {{"v", "3.0*y"}}, .expectedResult = {9.0}},
                                         (FormulaTestsVectorParameters){.formula = "t*vel + test", .nested = {{"vel", "3.0*y"}, {"test", "z"}}, .expectedResult = {27}},
                                         (FormulaTestsVectorParameters){.formula = "v*x, v*z + y", .nested = {{"v", "3.0*y"}}, .expectedResult = {6, 20}},
                                         (FormulaTestsVectorParameters){.formula = "0, i*y, t/i", .nested = {{"i", "10.0"}}, .expectedResult = {0, 20, 0.4}},
                                         (FormulaTestsVectorParameters){.formula = "0+CC, i*y+CC, t/i+AA",
                                                                        .nested = {{"i", "10.0"}},
                                                                        .constants = ablate::parameters::MapParameters::Create({{"CC", "3"}, {"AA", "1.5"}}),
                                                                        .expectedResult = {3, 23, 1.9}},
                                         (FormulaTestsVectorParameters){
                                             .formula = "0+CC, y+CC, t+AA", .constants = ablate::parameters::MapParameters::Create({{"CC", "3"}, {"AA", "1.5"}}), .expectedResult = {3, 5, 5.5}}));

}  // namespace ablateTesting::mathFunctions
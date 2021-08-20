#include <map>
#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "mathFunctions/parsedNested.hpp"
#include "parser/mockFactory.hpp"
#include "parser/registrar.hpp"

namespace ablateTesting::mathFunctions {

TEST(ParsedNestedTests, ShouldBeCreatedFromRegistar) {
    // arrange
    std::shared_ptr<ablateTesting::parser::MockFactory> mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    const std::string expectedClassType = "ablate::mathFunctions::ParsedSeries";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
    EXPECT_CALL(*mockFactory, Get(ablate::parser::ArgumentIdentifier<std::string>{.inputName = "formula"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("x+y+z+t"));

    std::shared_ptr<ablateTesting::parser::MockFactory> childFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    EXPECT_CALL(*mockFactory, GetFactory("nested")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(childFactory));
    EXPECT_CALL(*childFactory, GetKeys()).Times(::testing::Exactly(1));

    // act
    auto instance = ResolveAndCreate<ablate::mathFunctions::MathFunction>(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the ParsedNested";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::ParsedNested>(instance) != nullptr) << " should be an instance of ParsedNested";
}

TEST(ParsedNestedTests, ShouldThrowExceptionInvalidEquation) {
    // arrange/act/assert
    ASSERT_ANY_THROW(ablate::mathFunctions::ParsedNested("x+y+z+t+c", {}));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ParsedNestedTestsScalarParameters {
    std::string formula;
    std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> nested;
    double expectedResult;
};

class ParsedNestedTestsScalarFixture : public ::testing::TestWithParam<ParsedNestedTestsScalarParameters> {};

TEST_P(ParsedNestedTestsScalarFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedNested(param.formula, param.nested);

    // act/assert
    ASSERT_DOUBLE_EQ(param.expectedResult, function.Eval(1.0, 2.0, 3.0, 4.0));
}

TEST_P(ParsedNestedTestsScalarFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedNested(param.formula, param.nested);

    const double array1[3] = {1.0, 2.0, 3.0};

    // act/assert
    ASSERT_DOUBLE_EQ(param.expectedResult, function.Eval(array1, 3, 4.0));
}

INSTANTIATE_TEST_SUITE_P(ParsedNestedTests, ParsedNestedTestsScalarFixture,
                         testing::Values((ParsedNestedTestsScalarParameters){.formula = "v*x", .nested = {{"v", ablate::mathFunctions::Create(2.0)}}, .expectedResult = 2.0},
                                         (ParsedNestedTestsScalarParameters){.formula = "v*x + z", .nested = {{"v", ablate::mathFunctions::Create("3.0*y")}}, .expectedResult = 9.0},
                                         (ParsedNestedTestsScalarParameters){.formula = "t*vel + test",
                                                                             .nested = {{"vel", ablate::mathFunctions::Create("3.0*y")}, {"test", ablate::mathFunctions::Create("z")}},
                                                                             .expectedResult = 27}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ParsedNestedTestsVectorParameters {
    std::string formula;
    std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> nested;
    std::vector<double> expectedResult;
};

class ParsedNestedTestsVectorFixture : public ::testing::TestWithParam<ParsedNestedTestsVectorParameters> {};

TEST_P(ParsedNestedTestsVectorFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedNested(param.formula, param.nested);
    std::vector<double> result(param.expectedResult.size(), NAN);

    // act
    function.Eval(1.0, 2.0, 3.0, 4.0, result);

    // assert
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

TEST_P(ParsedNestedTestsVectorFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedNested(param.formula, param.nested);
    std::vector<double> result(param.expectedResult.size(), NAN);

    const double array[3] = {1.0, 2.0, 3.0};

    // act
    function.Eval(array, 3, 4, result);

    // assert
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

TEST_P(ParsedNestedTestsVectorFixture, ShouldComputeCorrectAnswerPetscFunction) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedNested(param.formula, param.nested);
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
INSTANTIATE_TEST_SUITE_P(ParsedNestedTests, ParsedNestedTestsVectorFixture,
                         testing::Values((ParsedNestedTestsVectorParameters){.formula = "v*x", .nested = {{"v", ablate::mathFunctions::Create(2.0)}}, .expectedResult = {2.0}},
                                         (ParsedNestedTestsVectorParameters){.formula = "v*x + z", .nested = {{"v", ablate::mathFunctions::Create("3.0*y")}}, .expectedResult = {9.0}},
                                         (ParsedNestedTestsVectorParameters){.formula = "t*vel + test",
                                                                             .nested = {{"vel", ablate::mathFunctions::Create("3.0*y")}, {"test", ablate::mathFunctions::Create("z")}},
                                                                             .expectedResult = {27}},
                                         (ParsedNestedTestsVectorParameters){.formula = "v*x, v*z + y", .nested = {{"v", ablate::mathFunctions::Create("3.0*y")}}, .expectedResult = {6, 20}},
                                         (ParsedNestedTestsVectorParameters){.formula = "0, i*y, t/i", .nested = {{"i", ablate::mathFunctions::Create(10.0)}}, .expectedResult = {0, 20, 0.4}}));

}  // namespace ablateTesting::mathFunctions
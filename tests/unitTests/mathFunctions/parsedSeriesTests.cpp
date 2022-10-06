#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/parsedSeries.hpp"
#include "mockFactory.hpp"
#include "parameters/mapParameters.hpp"
#include "registrar.hpp"

namespace ablateTesting::mathFunctions {

TEST(ParsedSeriesTests, ShouldBeCreatedFromRegistar) {
    // arrange
    std::shared_ptr<cppParserTesting::MockFactory> mockFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedClassType = "ablate::mathFunctions::ParsedSeries";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::string>{.inputName = "formula"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return("x+y+z+t"));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<int>{.inputName = "lowerBound"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(1));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<int>{.inputName = "upperBound"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(10));
    EXPECT_CALL(*mockFactory, Contains("constants")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));

    // act
    auto createMethod = Creator<ablate::mathFunctions::MathFunction>::GetCreateMethod(mockFactory->GetClassType());
    auto instance = createMethod(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the ParsedSeries";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::ParsedSeries>(instance) != nullptr) << " should be an instance of ParsedSeries";
}

TEST(ParsedSeriesTests, ShouldThrowExceptionInvalidEquation) {
    // arrange/act/assert
    ASSERT_THROW(ablate::mathFunctions::ParsedSeries("x+y+z+t+c"), std::invalid_argument);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ParsedSeriesTestsScalarParameters {
    std::string formula;
    int lowerBound;
    int upperBound;
    std::shared_ptr<ablate::parameters::Parameters> constants = nullptr;
    double expectedResult;
};

class ParsedSeriesTestsScalarFixture : public ::testing::TestWithParam<ParsedSeriesTestsScalarParameters> {};

TEST_P(ParsedSeriesTestsScalarFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedSeries(param.formula, param.lowerBound, param.upperBound, param.constants);

    // act/assert
    ASSERT_DOUBLE_EQ(param.expectedResult, function.Eval(1.0, 2.0, 3.0, 4.0));
}

TEST_P(ParsedSeriesTestsScalarFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedSeries(param.formula, param.lowerBound, param.upperBound, param.constants);

    const double array1[3] = {1.0, 2.0, 3.0};

    // act/assert
    ASSERT_DOUBLE_EQ(param.expectedResult, function.Eval(array1, 3, 4.0));
}

INSTANTIATE_TEST_SUITE_P(ParsedSeriesTests, ParsedSeriesTestsScalarFixture,
                         testing::Values((ParsedSeriesTestsScalarParameters){.formula = "i*x", .lowerBound = 1, .upperBound = 100, .expectedResult = 5050},
                                         (ParsedSeriesTestsScalarParameters){.formula = "i*x + y", .lowerBound = 0, .upperBound = 0, .expectedResult = 2},
                                         (ParsedSeriesTestsScalarParameters){.formula = "t*i*i", .lowerBound = 0, .upperBound = 10, .expectedResult = 1540},
                                         (ParsedSeriesTestsScalarParameters){
                                             .formula = "t*i*i*a", .lowerBound = 0, .upperBound = 10, .constants = ablate::parameters::MapParameters::Create({{"a", "0.5"}}), .expectedResult = 770}));

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ParsedSeriesTestsVectorParameters {
    std::string formula;
    int lowerBound;
    int upperBound;
    std::shared_ptr<ablate::parameters::Parameters> constants = nullptr;
    std::vector<double> expectedResult;
};

class ParsedSeriesTestsVectorFixture : public ::testing::TestWithParam<ParsedSeriesTestsVectorParameters> {};

TEST_P(ParsedSeriesTestsVectorFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedSeries(param.formula, param.lowerBound, param.upperBound, param.constants);
    std::vector<double> result(param.expectedResult.size(), NAN);

    // act
    function.Eval(1.0, 2.0, 3.0, 4.0, result);

    // assert
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

TEST_P(ParsedSeriesTestsVectorFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedSeries(param.formula, param.lowerBound, param.upperBound, param.constants);
    std::vector<double> result(param.expectedResult.size(), NAN);

    const double array[3] = {1.0, 2.0, 3.0};

    // act
    function.Eval(array, 3, 4, result);

    // assert
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

TEST_P(ParsedSeriesTestsVectorFixture, ShouldComputeCorrectAnswerPetscFunction) {
    // arrange
    const auto& param = GetParam();
    auto function = ablate::mathFunctions::ParsedSeries(param.formula, param.lowerBound, param.upperBound, param.constants);
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
INSTANTIATE_TEST_SUITE_P(ParsedSeriesTests, ParsedSeriesTestsVectorFixture,
                         testing::Values((ParsedSeriesTestsVectorParameters){.formula = "i*x", .lowerBound = 1, .upperBound = 100, .expectedResult = {5050}},
                                         (ParsedSeriesTestsVectorParameters){.formula = "i*x + y", .lowerBound = 0, .upperBound = 0, .expectedResult = {2}},
                                         (ParsedSeriesTestsVectorParameters){.formula = "t*i*i", .lowerBound = 0, .upperBound = 10, .expectedResult = {1540}},
                                         (ParsedSeriesTestsVectorParameters){.formula = "i*x, i*x + y", .lowerBound = 1, .upperBound = 100, .expectedResult = {5050, 5250}},
                                         (ParsedSeriesTestsVectorParameters){.formula = "0, i*y, t", .lowerBound = 1, .upperBound = 10, .expectedResult = {0, 110, 40}},
                                         (ParsedSeriesTestsVectorParameters){.formula = "b, i*y*a, t*c",
                                                                             .lowerBound = 1,
                                                                             .upperBound = 10,
                                                                             .constants = ablate::parameters::MapParameters::Create({{"c", "3"}, {"a", "0.5"}, {"b", "1.5"}}),
                                                                             .expectedResult = {15, 55, 120}}));

}  // namespace ablateTesting::mathFunctions
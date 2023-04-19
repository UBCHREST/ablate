#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/linear.hpp"
#include "mockFactory.hpp"
#include "registrar.hpp"

namespace ablateTesting::mathFunctions {

TEST(LinearTests, ShouldBeCreatedFromRegistar) {
    // arrange
    std::shared_ptr<cppParserTesting::MockFactory> mockFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedClassType = "ablate::mathFunctions::Linear";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::vector<double>>{.inputName = "startValues", .description = "", .optional = false}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<double>{0.0}));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::vector<double>>{.inputName = "endValues", .description = "", .optional = false}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<double>{1.0}));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<double>{.inputName = "start", .description = "", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(0.0));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<double>{.inputName = "end", .description = "", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(1.0));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<int>{.inputName = "dir", .description = "", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(0));

    // act
    auto createMethod = Creator<ablate::mathFunctions::MathFunction>::GetCreateMethod(mockFactory->GetClassType());
    auto instance = createMethod(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the MathFunction";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::Linear>(instance) != nullptr) << " should be an instance of Linear";
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct LinearTestsParameters {
    std::vector<double> startValue;
    std::vector<double> endValue;
    double start;
    double end;
    int dir;

    // Store the expected parameters
    std::vector<std::pair<std::array<double, 3>, std::vector<double>>> testValues;
};

class LinearTestsFixture : public ::testing::TestWithParam<LinearTestsParameters> {};

TEST_P(LinearTestsFixture, ShouldEvalToScalarFromXYZ) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::Linear(params.startValue, params.endValue, params.start, params.end, params.dir);

    // act/assert
    for (const auto& testValue : params.testValues) {
        ASSERT_DOUBLE_EQ(function.Eval(testValue.first[0], testValue.first[1], testValue.first[2], 0), testValue.second.front());
    }
}

TEST_P(LinearTestsFixture, ShouldEvalToScalarFromCoord) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::Linear(params.startValue, params.endValue, params.start, params.end, params.dir);

    // act/assert
    for (const auto& testValue : params.testValues) {
        ASSERT_DOUBLE_EQ(function.Eval(testValue.first.data(), 3, 0), testValue.second.front());
    }
}

TEST_P(LinearTestsFixture, ShouldEvalToVectorFromXYZ) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::Linear(params.startValue, params.endValue, params.start, params.end, params.dir);

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(testValue.second.size());

        function.Eval(testValue.first[0], testValue.first[1], testValue.first[2], 0, result);

        for (std::size_t i = 0; i < result.size(); ++i) {
            ASSERT_DOUBLE_EQ(result[i], testValue.second[i]) << "Should be correct for index " << i << " for input " << testValue.first[0] << ", " << testValue.first[1] << ", " << testValue.first[2];
        }
    }
}

TEST_P(LinearTestsFixture, ShouldEvalToVectorFromCoord) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::Linear(params.startValue, params.endValue, params.start, params.end, params.dir);

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(testValue.second.size());

        function.Eval(testValue.first.data(), 3, 0, result);

        for (std::size_t i = 0; i < result.size(); ++i) {
            ASSERT_DOUBLE_EQ(result[i], testValue.second[i]) << "Should be correct for index " << i << " for input " << testValue.first[0] << ", " << testValue.first[1] << ", " << testValue.first[2];
        }
    }
}

TEST_P(LinearTestsFixture, ShouldProvideAndFunctionWithPetscFunction) {
    // arrange
    auto& params = GetParam();
    auto function = std::make_shared<ablate::mathFunctions::Linear>(params.startValue, params.endValue, params.start, params.end, params.dir);

    auto context = function->GetContext();
    auto functionPointer = function->GetPetscFunction();

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(testValue.second.size());

        PetscErrorCode errorCode = functionPointer(3, NAN, testValue.first.data(), (PetscInt)result.size(), result.data(), context);
        ASSERT_EQ(0, errorCode);
        for (std::size_t i = 0; i < result.size(); ++i) {
            ASSERT_DOUBLE_EQ(result[i], testValue.second[i]) << "Should be correct for index " << i << " for input " << testValue.first[0] << ", " << testValue.first[1] << ", " << testValue.first[2];
        }
    }
}

INSTANTIATE_TEST_SUITE_P(LinearTests, LinearTestsFixture,
                         testing::Values((LinearTestsParameters){.startValue = {0, 10, -20},
                                                                 .endValue = {1, -10, 20},
                                                                 .start = 0.0,
                                                                 .end = 1.0,
                                                                 .dir = 0,
                                                                 .testValues =
                                                                     {
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, 0.0}, {0.0, 10.0, -20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({-1, 0.0, 0.0}, {0.0, 10.0, -20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({1.0, 0.0, 0.0}, {1.0, -10.0, 20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({1.2, 0.0, 0.0}, {1.0, -10.0, 20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.5, 0.0, 0.0}, {.5, 0, 0}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.25, 0.0, 0.0}, {.25, 5, -10}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.75, 0.0, 0.0}, {.75, -5, 10}),

                                                                     }},
                                         (LinearTestsParameters){.startValue = {0, 10, -20},
                                                                 .endValue = {1, -10, 20},
                                                                 .start = 1,
                                                                 .end = 1.5,
                                                                 .dir = 1,
                                                                 .testValues =
                                                                     {
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 1.0, 0.0}, {0.0, 10.0, -20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0, 0.0, 0.0}, {0.0, 10.0, -20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 1.5, 0.0}, {1.0, -10.0, 20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 2, 0.0}, {1.0, -10.0, 20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 1.25, 0.0}, {.5, 0, 0}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 1.125, 0.0}, {.25, 5, -10}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 1.375, 0.0}, {.75, -5, 10}),

                                                                     }},
                                         (LinearTestsParameters){.startValue = {0, 10, -20},
                                                                 .endValue = {1, -10, 20},
                                                                 .start = -1.0,
                                                                 .end = -0.5,
                                                                 .dir = 2,
                                                                 .testValues =
                                                                     {
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, -1.0}, {0.0, 10.0, -20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0, 0.0, -2}, {0.0, 10.0, -20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, -0.5}, {1.0, -10.0, 20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, 0.0}, {1.0, -10.0, 20}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, -.75}, {.5, 0, 0}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 0.0, -0.875}, {.25, 5, -10}),
                                                                         std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 0.0, -0.625}, {.75, -5, 10}),
                                                                     }}),
                         [](const testing::TestParamInfo<LinearTestsParameters>& info) { return "linearTest" + std::to_string(info.index); });

}  // namespace ablateTesting::mathFunctions
#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/peak.hpp"
#include "mockFactory.hpp"
#include "registrar.hpp"

namespace ablateTesting::mathFunctions {

TEST(PeakTests, ShouldBeCreatedFromRegistar) {
    // arrange
    std::shared_ptr<cppParserTesting::MockFactory> mockFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedClassType = "ablate::mathFunctions::Peak";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::vector<double>>{.inputName = "startValues"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(std::vector<double>{0.0}));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::vector<double>>{.inputName = "peakValues"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(std::vector<double>{0.5}));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::vector<double>>{.inputName = "endValues"})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(std::vector<double>{1.0}));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<double>{.inputName = "start", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(0.0));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<double>{.inputName = "peak", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(1.0));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<double>{.inputName = "end", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(1.0));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<int>{.inputName = "dir", .optional = true})).Times(::testing::Exactly(1)).WillOnce(::testing::Return(0));

    // act
    auto createMethod = Creator<ablate::mathFunctions::MathFunction>::GetCreateMethod(mockFactory->GetClassType());
    auto instance = createMethod(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the MathFunction";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::Peak>(instance) != nullptr) << " should be an instance of Peak";
}
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct PeakTestsParameters {
    std::vector<double> startValue;
    std::vector<double> peakValue;
    std::vector<double> endValue;
    double start;
    double peak;
    double end;
    int dir;

    // Store the expected parameters
    std::vector<std::pair<std::array<double, 3>, std::vector<double>>> testValues;
};

class PeakTestsFixture : public ::testing::TestWithParam<PeakTestsParameters> {};

TEST_P(PeakTestsFixture, ShouldEvalToScalarFromXYZ) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::Peak(params.startValue, params.peakValue, params.endValue, params.start, params.peak, params.end, params.dir);

    // act/assert
    for (const auto& testValue : params.testValues) {
        ASSERT_DOUBLE_EQ(function.Eval(testValue.first[0], testValue.first[1], testValue.first[2], 0), testValue.second.front());
    }
}

TEST_P(PeakTestsFixture, ShouldEvalToScalarFromCoord) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::Peak(params.startValue, params.peakValue, params.endValue, params.start, params.peak, params.end, params.dir);

    // act/assert
    for (const auto& testValue : params.testValues) {
        ASSERT_DOUBLE_EQ(function.Eval(testValue.first.data(), 3, 0), testValue.second.front());
    }
}

TEST_P(PeakTestsFixture, ShouldEvalToVectorFromXYZ) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::Peak(params.startValue, params.peakValue, params.endValue, params.start, params.peak, params.end, params.dir);

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(testValue.second.size());

        function.Eval(testValue.first[0], testValue.first[1], testValue.first[2], 0, result);

        for (std::size_t i = 0; i < result.size(); ++i) {
            ASSERT_NEAR(result[i], testValue.second[i], 1E-8) << "Should be correct for index " << i << " for input " << testValue.first[0] << ", " << testValue.first[1] << ", " << testValue.first[2];
        }
    }
}

TEST_P(PeakTestsFixture, ShouldEvalToVectorFromCoord) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::Peak(params.startValue, params.peakValue, params.endValue, params.start, params.peak, params.end, params.dir);

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(testValue.second.size());

        function.Eval(testValue.first.data(), 3, 0, result);

        for (std::size_t i = 0; i < result.size(); ++i) {
            ASSERT_NEAR(result[i], testValue.second[i], 1E-8) << "Should be correct for index " << i << " for input " << testValue.first[0] << ", " << testValue.first[1] << ", " << testValue.first[2];
        }
    }
}

TEST_P(PeakTestsFixture, ShouldProvideAndFunctionWithPetscFunction) {
    // arrange
    auto& params = GetParam();
    auto function = std::make_shared<ablate::mathFunctions::Peak>(params.startValue, params.peakValue, params.endValue, params.start, params.peak, params.end, params.dir);

    auto context = function->GetContext();
    auto functionPointer = function->GetPetscFunction();

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(testValue.second.size());

        PetscErrorCode errorCode = functionPointer(3, NAN, testValue.first.data(), (PetscInt)result.size(), result.data(), context);
        ASSERT_EQ(0, errorCode);
        for (std::size_t i = 0; i < result.size(); ++i) {
            ASSERT_NEAR(result[i], testValue.second[i], 1E-8) << "Should be correct for index " << i << " for input " << testValue.first[0] << ", " << testValue.first[1] << ", " << testValue.first[2];
        }
    }
}

INSTANTIATE_TEST_SUITE_P(LinearTests, PeakTestsFixture,
                         testing::Values((PeakTestsParameters){.startValue = {0, 10, -20},
                                                               .peakValue = {1, -10, 20},
                                                               .endValue = {.5, 0.0, 20},
                                                               .start = 0.0,
                                                               .peak = 1.0,
                                                               .end = 3.0,
                                                               .dir = 0,
                                                               .testValues = {std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, 0.0}, {0.0, 10.0, -20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({-1, 0.0, 0.0}, {0.0, 10.0, -20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({1.0, 0.0, 0.0}, {1.0, -10.0, 20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({0.5, 0.0, 0.0}, {.5, 0, 0}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({0.25, 0.0, 0.0}, {.25, 5, -10}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({0.75, 0.0, 0.0}, {.75, -5, 10}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({1.5, 0.0, 0.0}, {0.875, -7.5, 20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({2.5, 0.0, 0.0}, {0.625, -2.5, 20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({3.0, 0.0, 0.0}, {.5, 0.0, 20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({3.5, 0.0, 0.0}, {.5, 0.0, 20})

                                                               }},
                                         (PeakTestsParameters){.startValue = {0, 10, -20},
                                                               .peakValue = {1, -10, 20},
                                                               .endValue = {.5, 0.0, 20},
                                                               .start = 1,
                                                               .peak = 1.5,
                                                               .end = 3.0,
                                                               .dir = 1,
                                                               .testValues = {std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 1.0, 0.0}, {0.0, 10.0, -20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({0, 0.0, 0.0}, {0.0, 10.0, -20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 1.5, 0.0}, {1.0, -10.0, 20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 1.25, 0.0}, {.5, 0, 0}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 1.125, 0.0}, {.25, 5, -10}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 1.375, 0.0}, {.75, -5, 10}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 2.1, 0.0}, {0.8, -6, 20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 2.7, 0.0}, {0.6, -2, 20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 3.0, 0.0}, {.5, 0.0, 20}),
                                                                              std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 3.5, NAN}, {.5, 0.0, 20})

                                                               }},
                                         (PeakTestsParameters){.startValue = {0, 10, -20},
                                                               .peakValue = {0.5, 0.0, 0.0},
                                                               .endValue = {1, -10, 20},
                                                               .start = -1.0,
                                                               .peak = -0.75,
                                                               .end = -0.5,
                                                               .dir = 2,
                                                               .testValues = {
                                                                   std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, -1.0}, {0.0, 10.0, -20}),
                                                                   std::make_pair<std::array<double, 3>, std::vector<double>>({0, 0.0, -2}, {0.0, 10.0, -20}),
                                                                   std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, -0.5}, {1.0, -10.0, 20}),
                                                                   std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, 0.0}, {1.0, -10.0, 20}),
                                                                   std::make_pair<std::array<double, 3>, std::vector<double>>({0.0, 0.0, -.75}, {.5, 0, 0}),
                                                                   std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 0.0, -0.875}, {.25, 5, -10}),
                                                                   std::make_pair<std::array<double, 3>, std::vector<double>>({NAN, 0.0, -0.625}, {.75, -5, 10}),
                                                               }}));

}  // namespace ablateTesting::mathFunctions
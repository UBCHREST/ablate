#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/nearestPoint.hpp"
#include "mockFactory.hpp"
#include "registrar.hpp"

namespace ablateTesting::mathFunctions {

TEST(NearestPointTests, ShouldBeCreatedFromRegistrar) {
    // arrange
    std::shared_ptr<cppParserTesting::MockFactory> mockFactory = std::make_shared<cppParserTesting::MockFactory>();
    const std::string expectedClassType = "ablate::mathFunctions::NearestPoint";
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::vector<double>>{.inputName = "coordinates", .description = "", .optional = false}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<double>{0.0}));
    EXPECT_CALL(*mockFactory, Get(cppParser::ArgumentIdentifier<std::vector<double>>{.inputName = "values", .description = "", .optional = false}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(std::vector<double>{0.0}));
    // act
    auto createMethod = Creator<ablate::mathFunctions::MathFunction>::GetCreateMethod(mockFactory->GetClassType());
    auto instance = createMethod(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the MathFunction";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::NearestPoint>(instance) != nullptr) << " should be an instance of NearestPoint";
}
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct NearestPointTestParameters {
    // list of coordinates (x1, y1, z1, x2, y2, etc.)
    std::vector<double> coordinates;
    // list of values in the same order as the coordinates
    std::vector<double> values;

    // Store the expected parameters
    std::vector<std::pair<std::array<double, 3>, double>> testValues;
};

class NearestPointTestFixture : public ::testing::TestWithParam<NearestPointTestParameters> {};

TEST_P(NearestPointTestFixture, ShouldEvalToScalarFromXYZ) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::NearestPoint(params.coordinates, params.values);

    // act/assert
    for (const auto& testValue : params.testValues) {
        ASSERT_DOUBLE_EQ(function.Eval(testValue.first[0], testValue.first[1], testValue.first[2], 0), testValue.second)
            << "for point (" << testValue.first[0] << ", " << testValue.first[1] << ", " << testValue.first[2] << ")";
    }
}

TEST_P(NearestPointTestFixture, ShouldEvalToScalarFromCoord) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::NearestPoint(params.coordinates, params.values);

    // act/assert
    for (const auto& testValue : params.testValues) {
        ASSERT_DOUBLE_EQ(function.Eval(testValue.first.data(), 3, 0), testValue.second);
    }
}

TEST_P(NearestPointTestFixture, ShouldEvalToVectorFromXYZ) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::NearestPoint(params.coordinates, params.values);

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(1);

        function.Eval(testValue.first[0], testValue.first[1], testValue.first[2], 0, result);

        ASSERT_DOUBLE_EQ(result[0], testValue.second);
    }
}

TEST_P(NearestPointTestFixture, ShouldEvalToVectorFromCoord) {
    // arrange
    auto& params = GetParam();
    auto function = ablate::mathFunctions::NearestPoint(params.coordinates, params.values);

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(1);

        function.Eval(testValue.first.data(), 3, 0, result);

        ASSERT_DOUBLE_EQ(result[0], testValue.second);
    }
}

TEST_P(NearestPointTestFixture, ShouldProvideAndFunctionWithPetscFunction) {
    // arrange
    auto& params = GetParam();
    auto function = std::make_shared<ablate::mathFunctions::NearestPoint>(params.coordinates, params.values);

    auto context = function->GetContext();
    auto functionPointer = function->GetPetscFunction();

    // act/assert
    for (const auto& testValue : params.testValues) {
        std::vector<double> result(1);

        PetscErrorCode errorCode = functionPointer(3, NAN, testValue.first.data(), (PetscInt)result.size(), result.data(), context);
        ASSERT_EQ(0, errorCode);
        ASSERT_DOUBLE_EQ(result[0], testValue.second);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PeakTets, NearestPointTestFixture,
    testing::Values(
        (NearestPointTestParameters){
            .coordinates = {0.5121333, 0.832685302, 0.273289879, 0.712122287, 0.692009831, 0.760666858, 0.223585321, 0.77322178, 0.248150353, 0.164191059, 0.198044227, 0.972994704},
            .values = {0.213380302, 0.976929891, 0.581266933, 0.558226789, 0.538619306, 0.278872823, 0.406111608, 0.220206646, 0.605487012, 0.071767326, 0.320414969, 0.736646384},
            .testValues = {std::make_pair<std::array<double, 3>, double>({0.2, 0.0, 0.0}, 0.320414969),
                           std::make_pair<std::array<double, 3>, double>({0.2, 0.2, 0.0}, 0.320414969),
                           std::make_pair<std::array<double, 3>, double>({0.0, 0.0, 0.0}, 0.071767326),
                           std::make_pair<std::array<double, 3>, double>({1.0, 0.0, 0.0}, 0.736646384),
                           std::make_pair<std::array<double, 3>, double>({.6, 0.0, 0.0}, 0.213380302)}},
        (NearestPointTestParameters){
            .coordinates = {0.5121333,   0.088040117, 0.832685302, 0.266172693, 0.273289879, 0.87316138,  0.712122287, 0.64332031, 0.692009831, 0.654407575, 0.760666858, 0.586392783,
                            0.223585321, 0.632429003, 0.77322178,  0.191583888, 0.248150353, 0.760530486, 0.164191059, 0.61304991, 0.198044227, 0.636913049, 0.972994704, 0.947179233},
            .values = {0.756883966, 0.729335788, 0.140049835, 0.727196538, 0.19118024, 0.54217176, 0.054379299, 0.878541279, 0.648211422, 0.525659988, 0.432148929, 0.390459105},
            .testValues = {std::make_pair<std::array<double, 3>, double>({0.6, 0.2, 0.0}, 0.756883966),
                           std::make_pair<std::array<double, 3>, double>({0.0, 0.7, 0.0}, 0.525659988),
                           std::make_pair<std::array<double, 3>, double>({.7, .7, 0.0}, 0.19118024),
                           std::make_pair<std::array<double, 3>, double>({-.7, -.7, 100.0}, 0.756883966)}},
        (NearestPointTestParameters){
            .coordinates = {0.5121333,   0.088040117, 0.902489559, 0.832685302, 0.266172693, 0.451770386, 0.273289879, 0.87316138,  0.219474717, 0.712122287, 0.64332031,  0.336543176,
                            0.692009831, 0.654407575, 0.156794693, 0.760666858, 0.586392783, 0.538728766, 0.223585321, 0.632429003, 0.512460568, 0.77322178,  0.191583888, 0.968408747,
                            0.248150353, 0.760530486, 0.203706594, 0.164191059, 0.61304991,  0.964650021, 0.198044227, 0.636913049, 0.416382878, 0.972994704, 0.947179233, 0.14745299},
            .values = {0.930256637, 0.643573381, 0.431167492, 0.40631711, 0.771787445, 0.274333382, 0.339731531, 0.868917216, 0.070364953, 0.96014164, 0.257556256, 0.014122445},
            .testValues = {std::make_pair<std::array<double, 3>, double>({-.7, -.7, -.7}, 0.257556256),
                           std::make_pair<std::array<double, 3>, double>({.5, .5, .5}, 0.274333382),
                           std::make_pair<std::array<double, 3>, double>({.5, .2, .1}, 0.643573381),
                           std::make_pair<std::array<double, 3>, double>({.1, -.2, .7}, 0.930256637)

            }}

        ),
    [](const testing::TestParamInfo<NearestPointTestParameters>& info) { return "nearestPointTest_" + std::to_string(info.param.coordinates.size() / info.param.values.size()) + "D"; });

}  // namespace ablateTesting::mathFunctions
#include <functional>
#include <memory>
#include "PetscTestFixture.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/constantValue.hpp"
#include "mathFunctions/geom/box.hpp"
#include "mathFunctions/geom/geometry.hpp"
#include "mathFunctions/geom/sphere.hpp"
#include "mathFunctions/geom/surface.hpp"

using namespace ablate::mathFunctions::geom;
namespace ablateTesting::mathFunctions::geom {

struct GeometryTestScalarParameters {
    std::function<std::shared_ptr<Geometry>()> createGeom;
    std::vector<PetscReal> xyz;
    double expectedResult;
};

class GeometryTestScalarFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<GeometryTestScalarParameters> {};

TEST_P(GeometryTestScalarFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();
    auto& xyz = param.xyz;

    double x = xyz[0];
    double y = xyz.size() > 1 ? xyz[1] : NAN;
    double z = xyz.size() > 2 ? xyz[2] : NAN;

    // act/assert
    ASSERT_DOUBLE_EQ(param.expectedResult, function->Eval(x, y, z, NAN));
}

TEST_P(GeometryTestScalarFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();

    // act/assert
    ASSERT_DOUBLE_EQ(param.expectedResult, function->Eval(&param.xyz[0], param.xyz.size(), NAN));
}

INSTANTIATE_TEST_SUITE_P(
    GeometryTests, GeometryTestScalarFixture,
    testing::Values(
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{.2, .2}, .1, std::vector<double>{10});
                                           },
                                       .xyz = {.25, .25},
                                       .expectedResult = 10},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{.2, .2}, .1, std::vector<double>{10});
                                           },
                                       .xyz = {.3, .3},
                                       .expectedResult = 0},
        (GeometryTestScalarParameters){.createGeom = []() { return std::make_shared<Sphere>(std::vector<double>{0.0}, .1, std::vector<double>{10}); }, .xyz = {.1}, .expectedResult = 10},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{-1, -2, -3}, 2, std::vector<double>{10}, std::vector<double>{4.2});
                                           },
                                       .xyz = {10, 11, 12},
                                       .expectedResult = 4.2},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{0.0, 0.0, 1.99}, 2, std::vector<double>{20}, std::vector<double>{4.2});
                                           },
                                       .xyz = {0.0, 0.0, 2.0},
                                       .expectedResult = 20},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1}, std::vector<double>{.3, .2}, std::vector<double>{10});
                                           },
                                       .xyz = {.25, .15},
                                       .expectedResult = 10},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1}, std::vector<double>{.3, .2}, std::vector<double>{10});
                                           },
                                       .xyz = {.15, .15},
                                       .expectedResult = 0},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1}, std::vector<double>{.3, .2}, std::vector<double>{10}, std::vector<double>{4.2});
                                           },
                                       .xyz = {.25, .25},
                                       .expectedResult = 4.2},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1}, std::vector<double>{10}, std::vector<double>{4.2});
                                           },
                                       .xyz = {-1.5, -1.0000001},
                                       .expectedResult = 10},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1});
                                           },
                                       .xyz = {-1.5, -1.0000001, 2},
                                       .expectedResult = 1},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1});
                                           },
                                       .xyz = {-1.5, -2.0000001, 2},
                                       .expectedResult = 0},
        (GeometryTestScalarParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step"); }, .xyz = {0.0, 0.0, 0.0}, .expectedResult = 1},
        (GeometryTestScalarParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step"); },
                                       .xyz = {0.0, 0.041, 0.0}, /**41 mm should be outside ***/
                                       .expectedResult = 0},
        (GeometryTestScalarParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step", std::vector<double>{10}, std::vector<double>{4.2}); },
                                       .xyz = {0.005, 0.035, 0.005},
                                       .expectedResult = 10},
        (GeometryTestScalarParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step", std::vector<double>{10}, std::vector<double>{4.2}); },
                                       .xyz = {0.01, 0.035, 0.01},
                                       .expectedResult = 4.2}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct GeometryTestVectorParameters {
    std::function<std::shared_ptr<Geometry>()> createGeom;
    std::vector<PetscReal> xyz;
    std::vector<double> expectedResult;
};

class GeometryTestVectorFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<GeometryTestVectorParameters> {};

TEST_P(GeometryTestVectorFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();
    std::vector<double> result(param.expectedResult.size(), NAN);
    auto& xyz = param.xyz;

    double x = xyz[0];
    double y = xyz.size() > 1 ? xyz[1] : NAN;
    double z = xyz.size() > 2 ? xyz[2] : NAN;

    // act
    function->Eval(x, y, z, NAN, result);

    // assert
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

TEST_P(GeometryTestVectorFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();
    std::vector<double> result(param.expectedResult.size(), NAN);

    // act
    function->Eval(&param.xyz[0], param.xyz.size(), NAN, result);

    // assert
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

TEST_P(GeometryTestVectorFixture, ShouldComputeCorrectAnswerPetscFunction) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();
    std::vector<double> result(param.expectedResult.size(), NAN);

    auto context = function->GetContext();
    auto functionPointer = function->GetPetscFunction();

    // act
    auto errorCode = functionPointer(param.xyz.size(), NAN, &param.xyz[0], result.size(), &result[0], context);

    // assert
    ASSERT_EQ(errorCode, 0);
    for (std::size_t i = 0; i < param.expectedResult.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedResult[i], result[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    GeometryTests, GeometryTestVectorFixture,
    testing::Values(
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{.2, .2}, .1, std::vector<double>{10});
                                           },
                                       .xyz = {.25, .25},
                                       .expectedResult = {10}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{.2, .2}, .1, std::vector<double>{12, 13, 14});
                                           },
                                       .xyz = {.3, .3},
                                       .expectedResult = {0.0, 0.0, 0.0}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{0.0}, .1, std::vector<double>{12, 13, 14});
                                           },
                                       .xyz = {.1},
                                       .expectedResult = {12.0, 13.0, 14.0}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{-1, -2, -3}, 2, std::vector<double>{10}, std::vector<double>{4.2, 6.2});
                                           },
                                       .xyz = {10, 11, 12},
                                       .expectedResult = {4.2, 6.2}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{0.0, 0.0, 1.99}, 2, std::vector<double>{20, 13}, std::vector<double>{4.2, 4.2});
                                           },
                                       .xyz = {0.0, 0.0, 2.0},
                                       .expectedResult = {20, 13}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1}, std::vector<double>{.3, .2}, std::vector<double>{10, 13});
                                           },
                                       .xyz = {.25, .15},
                                       .expectedResult = {10, 13}},
        (GeometryTestVectorParameters){
            .createGeom = []() { return std::make_shared<Box>(std::vector<double>{.2}, std::vector<double>{.3}, std::vector<double>{10}); }, .xyz = {.15}, .expectedResult = {0}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1, -.1}, std::vector<double>{.3, .2, .2}, std::vector<double>{10}, std::vector<double>{4.2, .23});
                                           },
                                       .xyz = {.25, .25, .25},
                                       .expectedResult = {4.2, .23}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1}, std::vector<double>{10, 11}, std::vector<double>{4.2});
                                           },
                                       .xyz = {-1.5, -1.0000001},
                                       .expectedResult = {10, 11}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1}, std::vector<double>{1, 2});
                                           },
                                       .xyz = {-1.5, -1.0000001, 2},
                                       .expectedResult = {1, 2}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1}, std::vector<double>{3, 3, 3, 3}, std::vector<double>{0, 1, 2, 3});
                                           },
                                       .xyz = {-1.5, -2.0000001, 2},
                                       .expectedResult = {0, 1, 2, 3}},
        (GeometryTestVectorParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step"); }, .xyz = {0.0, 0.0, 0.0}, .expectedResult = {1}},
        (GeometryTestVectorParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step"); },
                                       .xyz = {0.0, 0.041, 0.0}, /**41 mm should be outside ***/
                                       .expectedResult = {0}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step", std::vector<double>{10, 23, 22, 11}, std::vector<double>{0, 1, 2, 3});
                                           },
                                       .xyz = {0.005, 0.035, 0.005},
                                       .expectedResult = {10, 23, 22, 11}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step", std::vector<double>{4.2, 0.0}, std::vector<double>{1, 2});
                                           },
                                       .xyz = {0.01, 0.035, 0.01},
                                       .expectedResult = {1, 2}}));

}  // namespace ablateTesting::mathFunctions::geom
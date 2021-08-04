#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "mathFunctions/linearInterpolator.hpp"

namespace ablateTesting::mathFunctions {

static int ToXFunction(int dim, double time, const double x[], int nf, double* u, void* ctx) {
    u[0] = x[0];
    return 0;
}

TEST(LinearInterpolatorTests, ShouldCreateAndParseStream) {
    // arrange
    std::string csvFileString =
        "x, y, z\n"
        ".1, 2.2, 3\n"
        ".2, 2.2, 4\n"
        ".3, 1.1, 2\n";

    std::istringstream csvFileStream(csvFileString);

    // act
    ablate::mathFunctions::LinearTable interpolator(csvFileStream, "x", {"z", "y"}, ablate::mathFunctions::Create(ToXFunction));

    // assert
    auto expectedXValues = std::vector<double>{0.1, 0.2, 0.3};
    ASSERT_EQ(interpolator.GetXValues(), expectedXValues);

    auto expectedYValues0 = std::vector<double>{3, 4, 2};
    ASSERT_EQ(interpolator.GetYValues()[0], expectedYValues0);

    auto expectedYValues1 = std::vector<double>{2.2, 2.2, 1.1};
    ASSERT_EQ(interpolator.GetYValues()[1], expectedYValues1);
}

TEST(LinearInterpolatorTests, ShouldThrowErrorForMissingXColumn) {
    // arrange
    std::string csvFileString =
        "xx, y, z\n"
        ".1, 2.2, 3\n"
        ".2, 2.2, 4\n"
        ".3, 1.1, 2\n";

    std::istringstream csvFileStream(csvFileString);

    // act
    // assert
    ASSERT_THROW(ablate::mathFunctions::LinearTable(csvFileStream, "x", {"z", "y"}, ablate::mathFunctions::Create(ToXFunction)), std::invalid_argument);
}

TEST(LinearInterpolatorTests, ShouldThrowErrorForMissingYColumn) {
    // arrange
    std::string csvFileString =
        "x, yy, z\n"
        ".1, 2.2, 3\n"
        ".2, 2.2, 4\n"
        ".3, 1.1, 2\n";

    std::istringstream csvFileStream(csvFileString);

    // act
    // assert
    ASSERT_THROW(ablate::mathFunctions::LinearTable(csvFileStream, "x", {"z", "y"}, ablate::mathFunctions::Create(ToXFunction)), std::invalid_argument);
}

struct LinearInterpolatorTestParameters {
    std::vector<std::string> yColumns;
    std::vector<PetscReal> xyz;
    PetscReal time;
    std::shared_ptr<ablate::mathFunctions::MathFunction> xCoordFunction;
    std::vector<double> expectedValues;
};

class LinearInterpolatorTestFixture : public ::testing::TestWithParam<LinearInterpolatorTestParameters> {
   public:
    const std::string csvFileString =
        "x, y, z\n"
        ".1, 2.2, 3\n"
        ".2, 2.2, 4\n"
        ".3, 1.1, 2\n"
        ".5, 1.10, 2.4\n"
        ".55, 0.9, 2.4\n"
        ".6, -.1, 2.\n"
        ".7, 1.1, 2.\n";
};

TEST_P(LinearInterpolatorTestFixture, ShouldInterpolateValueUsingXYZTSignature) {
    // arrange
    std::istringstream csvFileStream(csvFileString);

    ablate::mathFunctions::LinearTable linearInterpolator(csvFileStream, "x", GetParam().yColumns, GetParam().xCoordFunction);

    // get the needed values
    double x = GetParam().xyz[0];
    double y = GetParam().xyz.size() > 1 ? GetParam().xyz[1] : 0.0;
    double z = GetParam().xyz.size() > 2 ? GetParam().xyz[2] : 0.0;

    // act
    auto value = linearInterpolator.Eval(x, y, z, GetParam().time);

    // assert
    ASSERT_DOUBLE_EQ(GetParam().expectedValues[0], value);
}

TEST_P(LinearInterpolatorTestFixture, ShouldInterpolateValueUsingXyzNdimsSignature) {
    // arrange
    std::istringstream csvFileStream(csvFileString);

    ablate::mathFunctions::LinearTable linearInterpolator(csvFileStream, "x", GetParam().yColumns, GetParam().xCoordFunction);

    // act
    auto value = linearInterpolator.Eval(&GetParam().xyz[0], GetParam().xyz.size(), GetParam().time);

    // assert
    ASSERT_DOUBLE_EQ(GetParam().expectedValues[0], value);
}

TEST_P(LinearInterpolatorTestFixture, ShouldInterpolateValueUsingXYZTVectorSignature) {
    // arrange
    std::istringstream csvFileStream(csvFileString);

    ablate::mathFunctions::LinearTable linearInterpolator(csvFileStream, "x", GetParam().yColumns, GetParam().xCoordFunction);
    std::vector<double> result(GetParam().expectedValues.size());

    // get the needed values
    double x = GetParam().xyz[0];
    double y = GetParam().xyz.size() > 1 ? GetParam().xyz[1] : 0.0;
    double z = GetParam().xyz.size() > 2 ? GetParam().xyz[2] : 0.0;

    // act
    linearInterpolator.Eval(x, y, z, GetParam().time, result);

    // assert
    for (std::size_t i = 0; i < result.size(); i++) {
        ASSERT_DOUBLE_EQ(GetParam().expectedValues[i], result[i]);
    }
}

TEST_P(LinearInterpolatorTestFixture, ShouldInterpolateValueUsingXyzNdimsTVectorSignature) {
    // arrange
    std::istringstream csvFileStream(csvFileString);

    ablate::mathFunctions::LinearTable linearInterpolator(csvFileStream, "x", GetParam().yColumns, GetParam().xCoordFunction);
    std::vector<double> result(GetParam().expectedValues.size());

    // act
    linearInterpolator.Eval(&GetParam().xyz[0], GetParam().xyz.size(), GetParam().time, result);

    // assert
    for (std::size_t i = 0; i < result.size(); i++) {
        ASSERT_DOUBLE_EQ(GetParam().expectedValues[i], result[i]);
    }
}

TEST_P(LinearInterpolatorTestFixture, ShouldInterpolateValueUsingPetscFunction) {
    // arrange
    std::istringstream csvFileStream(csvFileString);

    ablate::mathFunctions::LinearTable linearInterpolator(csvFileStream, "x", GetParam().yColumns, GetParam().xCoordFunction);
    std::vector<double> result(GetParam().expectedValues.size());

    auto petscFunction = linearInterpolator.GetPetscFunction();
    auto context = linearInterpolator.GetContext();

    // act
    petscFunction(GetParam().xyz.size(), GetParam().time, &GetParam().xyz[0], result.size(), &result[0], context);

    // assert
    for (std::size_t i = 0; i < result.size(); i++) {
        ASSERT_DOUBLE_EQ(GetParam().expectedValues[i], result[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(PerfectGasEOSTests, LinearInterpolatorTestFixture,
                         testing::Values(
                             (LinearInterpolatorTestParameters){
                                 .yColumns = {"y"},
                                 .xyz = {.3, NAN, NAN},
                                 .time = NAN,
                                 .xCoordFunction = ablate::mathFunctions::Create(ToXFunction),
                                 .expectedValues = {1.1},

                             },
                             (LinearInterpolatorTestParameters){
                                 .yColumns = {"z"},
                                 .xyz = {NAN, NAN, NAN},
                                 .time = .5,
                                 .xCoordFunction = ablate::mathFunctions::Create("t"),
                                 .expectedValues = {2.4},
                             },
                             (LinearInterpolatorTestParameters){
                                 .yColumns = {"y"},
                                 .xyz = {NAN, NAN, NAN},
                                 .time = -10,
                                 .xCoordFunction = ablate::mathFunctions::Create("t"),
                                 .expectedValues = {2.2},
                             },
                             (LinearInterpolatorTestParameters){
                                 .yColumns = {"z"},
                                 .xyz = {NAN},
                                 .time = 2.0,
                                 .xCoordFunction = ablate::mathFunctions::Create("t"),
                                 .expectedValues = {2.0},
                             },
                             (LinearInterpolatorTestParameters){
                                 .yColumns = {"z"},
                                 .xyz = {.1, .2, .05},
                                 .time = 0.0,
                                 .xCoordFunction = ablate::mathFunctions::Create("x + y + z"),
                                 .expectedValues = {2.1},
                             },
                             (LinearInterpolatorTestParameters){
                                 .yColumns = {"y", "z"},
                                 .xyz = {.1, .2, .05},
                                 .time = 0.0,
                                 .xCoordFunction = ablate::mathFunctions::Create("x + y + z + 0.21"),
                                 .expectedValues = {0.7, 2.32},
                             }),
                         [](const testing::TestParamInfo<LinearInterpolatorTestParameters>& info) { return std::to_string(info.index); });

}  // namespace ablateTesting::mathFunctions
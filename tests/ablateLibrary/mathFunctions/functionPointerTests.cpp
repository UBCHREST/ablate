#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/functionPointer.hpp"

namespace ablateTesting::mathFunctions {

TEST(FunctionPointerTests, ShouldEvalToScalarFromXYZ) {
    // arrange
    auto function = ablate::mathFunctions::FunctionPointer([](auto dim, auto time, auto loc, auto nf, auto u, auto ctx) {
        u[0] = loc[0] + loc[1] + loc[2] + time;
        return 0;
    });

    // act/assert
    ASSERT_DOUBLE_EQ(10.0, function.Eval(1.0, 2.0, 3.0, 4.0));
    ASSERT_DOUBLE_EQ(18.0, function.Eval(9.0, 2.0, 3.0, 4.0));
    ASSERT_DOUBLE_EQ(0.0, function.Eval(5.0, 2.0, 3.0, -10.0));
}

TEST(FunctionPointerTests, ShouldEvalToScalarFromCoord) {
    // arrange
    auto function = ablate::mathFunctions::FunctionPointer([](auto dim, auto time, auto loc, auto nf, auto u, auto ctx) {
        switch (dim) {
            case 3:
                u[0] = loc[0] + loc[1] + loc[2] + time;
                break;
            case 2:
                u[0] = loc[0] + loc[1] + time;
                break;
        }
        return 0;
    });

    // act/assert
    const double array1[3] = {1.0, 2.0, 3.0};
    ASSERT_DOUBLE_EQ(10.0, function.Eval(array1, 3, 4.0));

    const double array2[2] = {1.0, 2.0};
    ASSERT_DOUBLE_EQ(7.0, function.Eval(array2, 2, 4.0));

    const double array3[3] = {5, 5, -5};
    ASSERT_DOUBLE_EQ(0.0, function.Eval(array3, 3, -5));
}

TEST(FunctionPointerTests, ShouldEvalToVectorFromXYZ) {
    // arrange
    auto function = ablate::mathFunctions::FunctionPointer([](auto dim, auto time, auto loc, auto nf, auto u, auto ctx) {
        if (dim == 3 && nf >= 3) {
            u[0] = loc[0] + loc[1] + loc[2] + time;
            u[1] = loc[0] * loc[1] * loc[2];
            u[2] = time;
        }
        return 0;
    });

    // act/assert
    std::vector<double> result1 = {0, 0, 0};
    function.Eval(1.0, 2.0, 3.0, 4.0, result1);
    ASSERT_DOUBLE_EQ(result1[0], 10.0);
    ASSERT_DOUBLE_EQ(result1[1], 6.0);
    ASSERT_DOUBLE_EQ(result1[2], 4.0);

    std::vector<double> result2 = {0, 0, 0, 4};
    function.Eval(2.0, 2.0, 3.0, 4.0, result2);
    ASSERT_DOUBLE_EQ(result2[0], 11.0);
    ASSERT_DOUBLE_EQ(result2[1], 12.0);
    ASSERT_DOUBLE_EQ(result2[2], 4.0);
    ASSERT_DOUBLE_EQ(result2[3], 4.0);
}

TEST(FunctionPointerTests, ShouldEvalToVectorFromCoord) {
    // arrange
    auto function = ablate::mathFunctions::FunctionPointer([](auto dim, auto time, auto loc, auto nf, auto u, auto ctx) {
        if (dim == 3 && nf >= 3) {
            u[0] = loc[0] + loc[1] + loc[2] + time;
            u[1] = loc[0] * loc[1] * loc[2];
            u[2] = time;
        }
        return 0;
    });

    // act/assert
    const double array1[3] = {1.0, 2.0, 3.0};
    std::vector<double> result1 = {0, 0, 0};
    function.Eval(array1, 3, 4.0, result1);
    ASSERT_DOUBLE_EQ(result1[0], 10.0);
    ASSERT_DOUBLE_EQ(result1[1], 6.0);
    ASSERT_DOUBLE_EQ(result1[2], 4.0);
}

TEST(FunctionPointerTests, ShouldProvideAndFunctionWithPetscFunctionWhenSimpleFunction) {
    // arrange
    auto function = ablate::mathFunctions::FunctionPointer([](auto dim, auto time, auto loc, auto nf, auto u, auto ctx) {
        u[0] = loc[0] + loc[1] + loc[2] + time;
        u[1] = loc[0] * loc[1] * loc[2];
        return 0;
    });

    auto context = function.GetContext();
    auto functionPointer = function.GetPetscFunction();

    const PetscReal x[3] = {1.0, 2.0, 3.0};
    PetscScalar result[2] = {0.0, 0.0};

    // act
    auto errorCode = functionPointer(3, 4.0, x, 2, result, context);

    // assert
    ASSERT_EQ(0, errorCode);
    ASSERT_DOUBLE_EQ(10.0, result[0]);
    ASSERT_DOUBLE_EQ(6.0, result[1]);
}
}  // namespace ablateTesting::mathFunctions
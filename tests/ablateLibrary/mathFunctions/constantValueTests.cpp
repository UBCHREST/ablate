#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/constantValue.hpp"

namespace ablateTesting::mathFunctions {

TEST(ConstantValueTests, ShouldEvalToScalarFromXYZ) {
    // arrange
    auto function = ablate::mathFunctions::ConstantValue(std::vector<double>{3.0});

    // act/assert
    ASSERT_DOUBLE_EQ(3.0, function.Eval(1.0, 2.0, 3.0, 4.0));
    ASSERT_DOUBLE_EQ(3.0, function.Eval(9.0, 2.0, 3.0, 4.0));
    ASSERT_DOUBLE_EQ(3.0, function.Eval(5.0, 2.0, 3.0, -10.0));
}

TEST(ConstantValueTests, ShouldEvalToScalarFromCoord) {
    // arrange
    auto function = ablate::mathFunctions::ConstantValue(std::vector<double>{4.0});

    // act/assert
    const double array1[3] = {1.0, 2.0, 3.0};
    ASSERT_DOUBLE_EQ(4.0, function.Eval(array1, 3, 4.0));

    const double array2[2] = {1.0, 2.0};
    ASSERT_DOUBLE_EQ(4.0, function.Eval(array2, 2, 4.0));

    const double array3[3] = {5, 5, -5};
    ASSERT_DOUBLE_EQ(4.0, function.Eval(array3, 3, -5));
}

TEST(ConstantValueTests, ShouldEvalToVectorFromXYZ) {
    // arrange
    auto function = ablate::mathFunctions::ConstantValue(std::vector<double>{10, 6.0, 4.0});

    // act/assert
    std::vector<double> result1 = {0, 0, 0};
    function.Eval(1.0, 2.0, 3.0, 4.0, result1);
    ASSERT_DOUBLE_EQ(result1[0], 10.0);
    ASSERT_DOUBLE_EQ(result1[1], 6.0);
    ASSERT_DOUBLE_EQ(result1[2], 4.0);

    std::vector<double> result2 = {0, 0, 0};
    function.Eval(2.0, 2.0, 3.0, 4.0, result2);
    ASSERT_DOUBLE_EQ(result2[0], 10.0);
    ASSERT_DOUBLE_EQ(result2[1], 6.0);
    ASSERT_DOUBLE_EQ(result2[2], 4.0);
}

TEST(ConstantValueTests, ShouldEvalToVectorFromCoord) {
    // arrange
    auto function = ablate::mathFunctions::ConstantValue(std::vector<double>{10, 6.0, 5.0});

    // act/assert
    const double array1[3] = {1.0, 2.0, 3.0};
    std::vector<double> result1 = {0, 0, 0};
    function.Eval(array1, 3, 4.0, result1);
    ASSERT_DOUBLE_EQ(result1[0], 10.0);
    ASSERT_DOUBLE_EQ(result1[1], 6.0);
    ASSERT_DOUBLE_EQ(result1[2], 5.0);
}

TEST(ConstantValueTests, ShouldProvideAndFunctionWithPetscFunction) {
    // arrange
    auto function = ablate::mathFunctions::ConstantValue(std::vector<double>{10, 6.0});


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
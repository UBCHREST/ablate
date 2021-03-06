#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/parsedFunction.hpp"
#include "parser/mockFactory.hpp"
#include "parser/registrar.hpp"

namespace ablateTesting::mathFunctions {

TEST(ParsedFunctionTests, ShouldBeCreatedFromRegistar) {
    // arrange
    std::shared_ptr<ablateTesting::parser::MockFactory> mockFactory = std::make_shared<ablateTesting::parser::MockFactory>();
    const std::string expectedClassType = "";  // should be default class
    EXPECT_CALL(*mockFactory, GetClassType()).Times(::testing::Exactly(1)).WillOnce(::testing::ReturnRef(expectedClassType));
    EXPECT_CALL(*mockFactory, Get(ablate::parser::ArgumentIdentifier<std::string>{"formula", "the formula that may accept x, y, z, t"}))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return("x+y+z+t"));

    // act
    auto instance = ResolveAndCreate<ablate::mathFunctions::MathFunction>(mockFactory);

    // assert
    ASSERT_TRUE(instance != nullptr) << " should create an instance of the Parameters";
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::ParsedFunction>(instance) != nullptr) << " should be an instance of FactoryParameters";
}

TEST(ParsedFunctionTests, ShouldThrowExceptionInvalidEquation) {
    // arrange/act/assert
    ASSERT_ANY_THROW(ablate::mathFunctions::ParsedFunction("x+y+z+t+c"));
}

TEST(ParsedFunctionTests, ShouldEvalToScalarFromXYZ) {
    // arrange
    auto function = ablate::mathFunctions::ParsedFunction("x+y+z+t");

    // act/assert
    ASSERT_DOUBLE_EQ(10.0, function.Eval(1.0, 2.0, 3.0, 4.0));
    ASSERT_DOUBLE_EQ(18.0, function.Eval(9.0, 2.0, 3.0, 4.0));
    ASSERT_DOUBLE_EQ(0.0, function.Eval(5.0, 2.0, 3.0, -10.0));
}

TEST(ParsedFunctionTests, ShouldEvalToScalarFromCoord) {
    // arrange
    auto function = ablate::mathFunctions::ParsedFunction("x+y+z+t");

    // act/assert
    const double array1[3] = {1.0, 2.0, 3.0};
    ASSERT_DOUBLE_EQ(10.0, function.Eval(array1, 3, 4.0));

    const double array2[2] = {1.0, 2.0};
    ASSERT_DOUBLE_EQ(7.0, function.Eval(array2, 2, 4.0));

    const double array3[3] = {5, 5, -5};
    ASSERT_DOUBLE_EQ(0.0, function.Eval(array3, 3, -5));
}

TEST(ParsedFunctionTests, ShouldEvalToVectorFromXYZ) {
    // arrange
    auto function = ablate::mathFunctions::ParsedFunction("x+y+z+t,x*y*z,t");

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

    std::vector<double> result3 = {0, 0};
    ASSERT_THROW(function.Eval(2.0, 2.0, 3.0, 4.0, result3), std::invalid_argument);
}

TEST(ParsedFunctionTests, ShouldEvalToVectorFromCoord) {
    // arrange
    auto function = ablate::mathFunctions::ParsedFunction("x+y+z+t,x*y*z,t");

    // act/assert
    const double array1[3] = {1.0, 2.0, 3.0};
    std::vector<double> result1 = {0, 0, 0};
    function.Eval(array1, 3, 4.0, result1);
    ASSERT_DOUBLE_EQ(result1[0], 10.0);
    ASSERT_DOUBLE_EQ(result1[1], 6.0);
    ASSERT_DOUBLE_EQ(result1[2], 4.0);

    const double array2[2] = {1.0, 2.0};
    std::vector<double> result2 = {0, 0, 0, 4};
    function.Eval(array2, 2, 4.0, result2);
    ASSERT_DOUBLE_EQ(result2[0], 7);
    ASSERT_DOUBLE_EQ(result2[1], 0);
    ASSERT_DOUBLE_EQ(result2[2], 4.0);
    ASSERT_DOUBLE_EQ(result2[3], 4.0);

    const double array3[3] = {1.0, 2.0, 3.0};
    std::vector<double> result3 = {0, 0};
    ASSERT_THROW(function.Eval(array3, 3, 4.0, result3), std::invalid_argument);
}

TEST(ParsedFunctionTests, ShouldProvideAndFunctionWithPetscFunction) {
    // arrange
    auto function = std::make_shared<ablate::mathFunctions::ParsedFunction>("x+y+z+t,x*y*z");

    auto context = function->GetContext();
    auto functionPointer = function->GetPetscFunction();

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
#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
namespace ablateTesting::mathFunctions {

TEST(FunctionFactoryTests, ShouldCreateFunctionForStatelessLambda) {
    // arrange
    auto function = ablate::mathFunctions::Create([](auto dim, auto time, auto loc, auto nf, auto u, auto ctx) {
        u[0] = loc[0] + loc[1] + loc[2] + time;
        return 0;
    });

    // act/assert
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::FunctionPointer>(function) != nullptr);
    ASSERT_DOUBLE_EQ(10.0, function->Eval(1.0, 2.0, 3.0, 4.0));
}

TEST(FunctionFactoryTests, ShouldCreateFunctionForStatefullLambda) {
    // arrange
    double offset;
    auto function = ablate::mathFunctions::Create([&offset](auto dim, auto time, auto loc, auto nf, auto u, auto ctx) {
        u[0] = loc[0] + loc[1] + loc[2] + time + offset;
        return 0;
    });
    offset = 2;

    // act/assert
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::FunctionWrapper>(function) != nullptr);
    ASSERT_DOUBLE_EQ(12.0, function->Eval(1.0, 2.0, 3.0, 4.0));
}

TEST(FunctionFactoryTests, ShouldCreateFunctionForParsedString) {
    // arrange
    auto function = ablate::mathFunctions::Create("x + y + z +t");

    // act/assert
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::SimpleFormula>(function) != nullptr);
    ASSERT_DOUBLE_EQ(10.0, function->Eval(1.0, 2.0, 3.0, 4.0));
}

static PetscErrorCode ExamplePetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx) {
    u[0] = x[0] + x[1] + x[2] + time;
    return 0;
}

TEST(FunctionFactoryTests, ShouldCreateFunctionFromFunction) {
    // arrange
    auto function = ablate::mathFunctions::Create(ExamplePetscFunction);

    // act/assert
    ASSERT_TRUE(std::dynamic_pointer_cast<ablate::mathFunctions::FunctionPointer>(function) != nullptr);
    ASSERT_DOUBLE_EQ(10.0, function->Eval(1.0, 2.0, 3.0, 4.0));
}

}  // namespace ablateTesting::mathFunctions
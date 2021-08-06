#include "eos/transport/constant.hpp"
#include "gtest/gtest.h"

TEST(ConstantTransportTests, ShouldRecordContantValues) {
    // ARRANGE
    const PetscReal expectedK = .234;
    const PetscReal expectedMu = 10.213;
    const PetscReal expectedDiff = 1.32;

    auto constantModel = std::make_shared<ablate::eos::transport::Constant>(expectedK, expectedMu, expectedDiff);

    // ACT
    PetscReal computedK = NAN;
    PetscReal computedMu = NAN;
    PetscReal computedDiff = NAN;

    constantModel->GetComputeConductivityFunction()(NAN, NULL, computedK, constantModel->GetComputeConductivityContext());
    constantModel->GetComputeViscosityFunction()(NAN, NULL, computedMu, constantModel->GetComputeViscosityContext());
    constantModel->GetComputeDiffusivityFunction()(NAN, NAN, NULL, computedDiff, constantModel->GetComputeDiffusivityContext());

    // ASSERT
    ASSERT_DOUBLE_EQ(expectedK, computedK);
    ASSERT_DOUBLE_EQ(computedMu, computedMu);
    ASSERT_DOUBLE_EQ(computedDiff, computedDiff);
}

TEST(ConstantTransportTests, ShouldReturnNullFunctionsIfAllValuesZero) {
    // ARRANGE
    auto constantModel = std::make_shared<ablate::eos::transport::Constant>();

    // ACT
    // ASSERT
    ASSERT_TRUE(constantModel->GetComputeConductivityFunction() == nullptr);
    ASSERT_TRUE(constantModel->GetComputeViscosityFunction() == nullptr);
    ASSERT_TRUE(constantModel->GetComputeDiffusivityFunction() == nullptr);
}
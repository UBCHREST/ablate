#include "gtest/gtest.h"
#include "utilities/mathUtilities.hpp"

struct TransformVectorTestParameters {
    std::vector<PetscScalar> normal;
    std::vector<PetscScalar> cartBasisVector;
    std::vector<PetscScalar> expectedNormBasisVector;
};

class TransformVectorTestFixture : public ::testing::TestWithParam<TransformVectorTestParameters> {};

TEST_P(TransformVectorTestFixture, ShouldTransformToAndFromNormalBasis) {
    // arrange
    PetscScalar transformMatrix[3][3] = {{NAN, NAN, NAN}, {NAN, NAN, NAN}, {NAN, NAN, NAN}};
    const auto dim = (PetscInt)GetParam().normal.size();

    // Compute the transformation matrix
    ablate::utilities::MathUtilities::ComputeTransformationMatrix(dim, &GetParam().normal[0], transformMatrix);

    // prepare a vector to hold the transformation matrix
    std::vector<PetscScalar> normBasisVectorCalc(GetParam().cartBasisVector.size(), NAN);
    std::vector<PetscScalar> cartBasisVectorCalc(GetParam().cartBasisVector.size(), NAN);

    // act
    ablate::utilities::MathUtilities::Multiply(dim, transformMatrix, &GetParam().cartBasisVector[0], &normBasisVectorCalc[0]);
    ablate::utilities::MathUtilities::MultiplyTranspose(dim, transformMatrix, &normBasisVectorCalc[0], &cartBasisVectorCalc[0]);

    // assert
    // The determinant of a transformation matrix should be 1.0
    if (dim == 1) {
        ASSERT_NEAR(1.0, PetscAbs(ablate::utilities::MathUtilities::ComputeDeterminant(dim, transformMatrix)), 1E-8) << "The determinant of a transformation matrix should be 1.0";
    } else {
        ASSERT_NEAR(1.0, ablate::utilities::MathUtilities::ComputeDeterminant(dim, transformMatrix), 1E-8) << "The determinant of a transformation matrix should be 1.0";
    }
    // the normal vector transformed in to the norm space should be {1, 0, 0}
    PetscScalar expectedNormVectorTransformed[3] = {1.0, 0.0, 0.0};
    std::vector<PetscScalar> normVectorTransformed(GetParam().cartBasisVector.size(), NAN);
    ablate::utilities::MathUtilities::Multiply(dim, transformMatrix, &GetParam().normal[0], &normVectorTransformed[0]);
    for (PetscInt d = 0; d < dim; d++) {
        ASSERT_NEAR(expectedNormVectorTransformed[d], normVectorTransformed[d], 1E-8) << "the normal vector transformed in to the norm space should be {1, 0, 0} in dir  " << d;
    }

    // The magnitude of the transformed vector and original should be the same
    ASSERT_NEAR(ablate::utilities::MathUtilities::MagVector(dim, &GetParam().cartBasisVector[0]), ablate::utilities::MathUtilities::MagVector(dim, &normBasisVectorCalc[0]), 1E-8)
        << "The magnitude of each vector should be the same";

    for (PetscInt d = 0; d < dim; d++) {
        ASSERT_NEAR(GetParam().expectedNormBasisVector[d], normBasisVectorCalc[d], 1E-8) << "The computed and expected norm basis vector should be equal in dir " << d;
        ASSERT_NEAR(GetParam().cartBasisVector[d], cartBasisVectorCalc[d], 1E-8) << "The Cartesian basis vector should be the same as specified in  " << d;
    }
}
INSTANTIATE_TEST_SUITE_P(MathUtilititiesTests, TransformVectorTestFixture,
                         testing::Values((TransformVectorTestParameters){.normal = {1.0}, .cartBasisVector = {.5}, .expectedNormBasisVector = {.5}},
                                         (TransformVectorTestParameters){.normal = {-1.0}, .cartBasisVector = {.5}, .expectedNormBasisVector = {-.5}},
                                         (TransformVectorTestParameters){.normal = {1.0}, .cartBasisVector = {-.5}, .expectedNormBasisVector = {-.5}},
                                         (TransformVectorTestParameters){.normal = {-1.0}, .cartBasisVector = {-.5}, .expectedNormBasisVector = {.5}},
                                         (TransformVectorTestParameters){.normal = {1.0, 0.0}, .cartBasisVector = {1.0, 0.0}, .expectedNormBasisVector = {1.0, 0.0}},
                                         (TransformVectorTestParameters){.normal = {0.0, 1.0}, .cartBasisVector = {1.0, 0.0}, .expectedNormBasisVector = {.0, -1.0}},
                                         (TransformVectorTestParameters){
                                             .normal = {1.0 / sqrt(2.0), 1.0 / sqrt(2.0)}, .cartBasisVector = {1.0, 0.0}, .expectedNormBasisVector = {1.0 / sqrt(2.0), -1.0 / sqrt(2.0)}},
                                         (TransformVectorTestParameters){.normal = {0.29814239699997197, 0.5962847939999439, 0.7453559924999299},
                                                                         .cartBasisVector = {.3, .4, -2.5},
                                                                         .expectedNormBasisVector = {-1.5354333445498556, 0.089442719099991547, 2.033333332675446}},
                                         (TransformVectorTestParameters){.normal = {1.0, 0.0, 0.0}, .cartBasisVector = {.3, .4, -2.5}, .expectedNormBasisVector = {.3, -.4, 2.5}},
                                         (TransformVectorTestParameters){.normal = {0.0, 1.0, 0.0}, .cartBasisVector = {.3, .4, -2.5}, .expectedNormBasisVector = {.4, .3, 2.5}},
                                         (TransformVectorTestParameters){.normal = {0.0, 0.0, 1.0}, .cartBasisVector = {.3, .4, -2.5}, .expectedNormBasisVector = {-2.5, -.3, -.4}},
                                         (TransformVectorTestParameters){.normal = {-1.0, 0.0, 0.0}, .cartBasisVector = {.3, .4, -2.5}, .expectedNormBasisVector = {-.3, +.4, 2.5}},
                                         (TransformVectorTestParameters){.normal = {0.0, -1.0, 0.0}, .cartBasisVector = {.3, .4, -2.5}, .expectedNormBasisVector = {-.4, -.3, 2.5}},
                                         (TransformVectorTestParameters){.normal = {0.0, 0.0, -1.0}, .cartBasisVector = {.3, .4, -2.5}, .expectedNormBasisVector = {2.5, .3, -.4}},
                                         (TransformVectorTestParameters){.normal = {0.23570226039552, 0.23570226039552, 0.94280904158206},
                                                                         .cartBasisVector = {.3, .4, -2.5},
                                                                         .expectedNormBasisVector = {-2.1920310216782859, -0.89738181263444372, -0.94324221828380639}}),
                         [](const testing::TestParamInfo<TransformVectorTestParameters>& info) { return std::to_string(info.param.normal.size()) + "_D_test" + std::to_string(info.index); });

struct CrossProductParameters {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> expectedResult;
};

class CrossProductFixture : public ::testing::TestWithParam<CrossProductParameters> {};

TEST_P(CrossProductFixture, ShouldComputeCrossProductDynamically) {
    // arrange
    std::vector<double> result(GetParam().expectedResult.size());

    // act
    ablate::utilities::MathUtilities::CrossVector(GetParam().a.size(), GetParam().a.data(), GetParam().b.data(), result.data());

    // assert
    for (std::size_t d = 0; d < GetParam().expectedResult.size(); ++d) {
        ASSERT_NEAR(GetParam().expectedResult[d], result[d], 1E-8) << "the cross product should be correct in dimension " << d;
    }
}

TEST_P(CrossProductFixture, ShouldComputeCrossProductStatically) {
    // arrange
    std::vector<double> result(GetParam().expectedResult.size());

    // act
    switch (GetParam().a.size()) {
        case 3:
            ablate::utilities::MathUtilities::CrossVector<3>(GetParam().a.data(), GetParam().b.data(), result.data());
            break;
        case 2:
            ablate::utilities::MathUtilities::CrossVector<2>(GetParam().a.data(), GetParam().b.data(), result.data());
            break;
        case 1:
            ablate::utilities::MathUtilities::CrossVector<1>(GetParam().a.data(), GetParam().b.data(), result.data());
            break;
        default:
            FAIL() << "Unknown vector dimension size";
    }

    // assert
    for (std::size_t d = 0; d < GetParam().expectedResult.size(); ++d) {
        ASSERT_NEAR(GetParam().expectedResult[d], result[d], 1E-8) << "the cross product should be correct in dimension " << d;
    }
}

INSTANTIATE_TEST_SUITE_P(MathUtilititiesTests, CrossProductFixture,
                         testing::Values((CrossProductParameters){.a = {1.0, 2.0, 3.0}, .b = {1.0, 5.0, 7.0}, .expectedResult = {-1.0, -4.0, 3.0}},
                                         (CrossProductParameters){.a = {-1.0, -2.0, 3.0}, .b = {4.0, 0.0, -8.0}, .expectedResult = {16.0, 4.0, 8.0}},
                                         (CrossProductParameters){.a = {.5, .1, -3.0}, .b = {.3, .1, -.1}, .expectedResult = {0.29, -0.85, 0.02}},
                                         (CrossProductParameters){.a = {1.0, 0., 0.0}, .b = {0.0, 0.0, 1.0}, .expectedResult = {0.0, -1.0, 0.0}},
                                         (CrossProductParameters){.a = {3.0, 5.0}, .b = {1.0, -1.0}, .expectedResult = {-8.0}},
                                         (CrossProductParameters){.a = {1.0}, .b = {.2}, .expectedResult = {0.0}}),
                         [](const testing::TestParamInfo<CrossProductParameters>& info) { return std::to_string(info.param.a.size()) + "_D_test" + std::to_string(info.index); });

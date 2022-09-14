#include <petsc.h>
#include <fstream>
#include "gtest/gtest.h"
#include "solver/dynamicRange.hpp"

namespace ablateTesting::solver {

class DynamicRangeTestFixture : public ::testing::TestWithParam<std::vector<PetscInt>> {};

TEST_P(DynamicRangeTestFixture, ShouldBuildAndUseRange) {
    // arrange
    ablate::solver::DynamicRange dynamicRange;

    for (const auto& i : GetParam()) {
        dynamicRange.Add(i);
    }

    // act
    const auto& range = dynamicRange.GetRange();

    // assert
    ASSERT_EQ(GetParam().size(), range.end - range.start);

    std::size_t refI = 0;
    for (PetscInt i = range.start; i < range.end; i++) {
        PetscInt index = range.points ? range.points[i] : i;

        // Assert Equal
        ASSERT_EQ(GetParam()[refI++], index) << "The expected index at " << i << " should be correct";
    }
}

INSTANTIATE_TEST_SUITE_P(MathUtilititiesTests, DynamicRangeTestFixture, testing::Values(std::vector<PetscInt>{2, 3, 6, 2, 5}, std::vector<PetscInt>{2}, std::vector<PetscInt>{}));
}  // namespace ablateTesting::solver

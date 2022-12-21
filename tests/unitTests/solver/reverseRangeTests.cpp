#include <petsc.h>
#include <fstream>
#include "gtest/gtest.h"
#include "solver/dynamicRange.hpp"
#include "solver/reverseRange.hpp"

namespace ablateTesting::solver {

class ReverseRangeTestFixture : public ::testing::TestWithParam<std::pair<ablate::solver::Range, std::vector<PetscInt>>> {};

TEST_P(ReverseRangeTestFixture, ShouldBuildAndUseRange) {
    // arrange
    auto range = GetParam().first;
    auto expectedAbsoluteIndex = GetParam().second;
    ablate::solver::ReverseRange reverseRange(range);

    // act/assert
    for (PetscInt index = range.start; index < range.end; ++index) {
        PetscInt point = range.GetPoint(index);

        // make sure we can reverse look up the index
        PetscInt reverseIndex = reverseRange.GetIndex(point);
        PetscInt reverseAbsoluteIndex = reverseRange.GetAbsoluteIndex(point);

        // Assert equal
        ASSERT_EQ(index, reverseIndex) << "The expected index at point " << point << " should be correct";
        ASSERT_EQ(expectedAbsoluteIndex[index - range.start], reverseAbsoluteIndex) << "The expected absolute index at point " << point << " should be correct";
    }
}

static PetscInt Test3[3] = {0, 1, 2};
static PetscInt Test4[3] = {4, 7, 9};
static PetscInt Test5[6] = {4, 7, 9, 11, 12, 17};

INSTANTIATE_TEST_SUITE_P(SolverTests, ReverseRangeTestFixture,
                         testing::Values(std::make_pair(ablate::solver::Range{.start = 0, .end = 3, .points = nullptr}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 3, .end = 6, .points = nullptr}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 0, .end = 3, .points = Test3}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 0, .end = 3, .points = Test4}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 3, .end = 6, .points = Test5}, std::vector<PetscInt>{0, 1, 2})));

}  // namespace ablateTesting::solver

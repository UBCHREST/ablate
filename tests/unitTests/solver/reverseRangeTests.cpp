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
static PetscInt Test6[2] = {-1000, -100};
static PetscInt Test7[1] = {3};
static PetscInt Test8[7] = {3, 4, 7, 8, 10, 11, 13};
static PetscInt Test9[6] = {3, 4, 11, 7, 13, 10};

INSTANTIATE_TEST_SUITE_P(SolverTests, ReverseRangeTestFixture,
                         testing::Values(std::make_pair(ablate::solver::Range{.start = 0, .end = 3, .points = nullptr}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 3, .end = 6, .points = nullptr}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 0, .end = 3, .points = Test3}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 0, .end = 3, .points = Test4}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 3, .end = 6, .points = Test5}, std::vector<PetscInt>{0, 1, 2}),
                                         std::make_pair(ablate::solver::Range{.start = 0, .end = 0, .points = Test6}, std::vector<PetscInt>{}),
                                         std::make_pair(ablate::solver::Range{.start = 1, .end = 1, .points = Test6}, std::vector<PetscInt>{}),
                                         std::make_pair(ablate::solver::Range{.start = 0, .end = 1, .points = Test7}, std::vector<PetscInt>{0}),
                                         std::make_pair(ablate::solver::Range{.start = 5, .end = 6, .points = Test8}, std::vector<PetscInt>{0}),
                                         std::make_pair(ablate::solver::Range{.start = 0, .end = 6, .points = Test9}, std::vector<PetscInt>{0, 1, 2, 3, 4, 5})));

}  // namespace ablateTesting::solver

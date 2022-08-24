#include <vector>
#include "gtest/gtest.h"
#include "io/interval/fixedInterval.hpp"

struct FixedIntervalParameters {
    int interval;
    std::vector<bool> expectedValues;
    std::vector<PetscReal> time;
    std::vector<PetscInt> steps;
};

class FixedIntervalTestFixture : public ::testing::TestWithParam<FixedIntervalParameters> {};

TEST_P(FixedIntervalTestFixture, ShouldProvideCorrectValuesAtCheck) {
    // arrange
    auto interval = std::make_shared<ablate::io::interval::FixedInterval>(GetParam().interval);

    // act/assert
    for (std::size_t i = 0; i < GetParam().expectedValues.size(); i++) {
        auto time = GetParam().time[i];
        auto step = GetParam().steps[i];

        ASSERT_EQ(GetParam().expectedValues[i], interval->Check(MPI_COMM_SELF, step, time));
    }
}

INSTANTIATE_TEST_SUITE_P(FixedIntervalTests, FixedIntervalTestFixture,
                         testing::Values((FixedIntervalParameters){.interval = 0, .expectedValues = {true, true, true, true}, .time = {NAN, NAN, NAN, NAN}, .steps = {0, 1, 2, 3}},
                                         (FixedIntervalParameters){.interval = 2, .expectedValues = {true, false, true, false, true}, .time = {NAN, NAN, NAN, NAN, NAN}, .steps = {0, 1, 2, 3, 4}},
                                         (FixedIntervalParameters){.interval = 3, .expectedValues = {true, false, true, false, true}, .time = {NAN, NAN, NAN, NAN, NAN}, .steps = {0, 1, 3, 5, 6}}),
                         [](const testing::TestParamInfo<FixedIntervalParameters>& info) { return "interval_" + std::to_string(info.param.interval); });

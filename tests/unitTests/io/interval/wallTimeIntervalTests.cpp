#include <chrono>
#include <deque>
#include <petscTestFixture.hpp>
#include <vector>
#include "gtest/gtest.h"
#include "io/interval/wallTimeInterval.hpp"

struct WallTimeIntervalParameters {
    long interval;
    std::vector<bool> expectedValues;
    std::deque<long> now;
};

class WallTimeIntervalTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<WallTimeIntervalParameters> {};

TEST_P(WallTimeIntervalTestFixture, ShouldProvideCorrectValuesAtCheck) {
    // arrange
    std::deque<long> nowList = GetParam().now;
    nowList.push_front(0);  // add init time
    auto nowFunction = [&nowList] {
        auto value = nowList.front();
        nowList.pop_front();
        return std::chrono::time_point<std::chrono::system_clock>() + std::chrono::seconds(value);
    };
    auto interval = std::make_shared<ablate::io::interval::WallTimeInterval>(GetParam().interval, nowFunction);

    // act/assert
    for (std::size_t i = 0; i < GetParam().expectedValues.size(); i++) {
        ASSERT_EQ(GetParam().expectedValues[i], interval->Check(MPI_COMM_SELF, -1, NAN)) << "at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(SimulationTimeIntervalTests, WallTimeIntervalTestFixture,
                         testing::Values((WallTimeIntervalParameters){.interval = 10, .expectedValues = {false, false, true, false, true, false}, .now = {0, 5, 10, 15, 20, 25}},
                                         (WallTimeIntervalParameters){.interval = 7, .expectedValues = {true, false, false, true, false, true}, .now = {7, 7, 10, 14, 12, 21}}),
                         [](const testing::TestParamInfo<WallTimeIntervalParameters>& info) { return "wallTimeInterval_" + std::to_string(info.param.interval); });

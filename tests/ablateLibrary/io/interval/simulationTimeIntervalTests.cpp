#include <vector>
#include "gtest/gtest.h"
#include "io/interval/simulationTimeInterval.hpp"

struct SimulationTimeIntervalParameters {
    double interval;
    std::vector<bool> expectedValues;
    std::vector<PetscReal> time;
    std::vector<PetscInt> steps;
};

class SimulationTimeIntervalTestFixture : public ::testing::TestWithParam<SimulationTimeIntervalParameters> {};

TEST_P(SimulationTimeIntervalTestFixture, ShouldProvideCorrectValuesAtCheck) {
    // arrange
    auto interval = std::make_shared<ablate::io::interval::SimulationTimeInterval>(GetParam().interval);

    // act/assert
    for (std::size_t i = 0; i < GetParam().expectedValues.size(); i++) {
        auto time = GetParam().time[i];
        auto step = GetParam().steps[i];

        ASSERT_EQ(GetParam().expectedValues[i], interval->Check(MPI_COMM_SELF, step, time)) << "with parameters time (" << time << ") step (" << step << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(
    SimulationTimeIntervalTests, SimulationTimeIntervalTestFixture,
    testing::Values((SimulationTimeIntervalParameters){.interval = 0.2, .expectedValues = {true, false, true, false, true}, .time = {.1, .2, .31, .41, .52}, .steps = {0, 0, 0, 0, 0}},
                    (SimulationTimeIntervalParameters){.interval = 0.2, .expectedValues = {true, false, true, false, false, true}, .time = {.1, .2, .31, .31, .41, .52}, .steps = {0, 0, 0, 0, 0, 0}}),
    [](const testing::TestParamInfo<SimulationTimeIntervalParameters>& info) { return "simulationTimeInterval_" + std::to_string(info.index); });

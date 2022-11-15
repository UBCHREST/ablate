#include <vector>
#include "gtest/gtest.h"
#include "io/interval/equationInterval.hpp"

struct EquationIntervalParameters {
    std::string equation;
    std::vector<bool> expectedValues;
    std::vector<PetscReal> time;
    std::vector<PetscInt> steps;
};

class EquationIntervalTestFixture : public ::testing::TestWithParam<EquationIntervalParameters> {};

TEST_P(EquationIntervalTestFixture, ShouldProvideCorrectValuesAtCheck) {
    // arrange
    auto interval = std::make_shared<ablate::io::interval::EquationInterval>(GetParam().equation);

    // act/assert
    for (std::size_t i = 0; i < GetParam().expectedValues.size(); i++) {
        auto time = GetParam().time[i];
        auto step = GetParam().steps[i];

        ASSERT_EQ(GetParam().expectedValues[i], interval->Check(MPI_COMM_SELF, step, time));
    }
}

INSTANTIATE_TEST_SUITE_P(
    EquationIntervalTests, EquationIntervalTestFixture,
    testing::Values((EquationIntervalParameters){.equation = "1", .expectedValues = {true, true, true, true}, .time = {0.0, .1, .2, .3}, .steps = {0, 1, 2, 3}},
                    (EquationIntervalParameters){.equation = "0", .expectedValues = {false, false, false, false}, .time = {0.0, .1, .2, .3}, .steps = {0, 1, 2, 3}},
                    (EquationIntervalParameters){.equation = "time >= .2", .expectedValues = {false, false, true, true}, .time = {0.0, .1, .2, .3}, .steps = {0, 1, 2, 3}},
                    (EquationIntervalParameters){.equation = "(time >= .1) && (time < .3)", .expectedValues = {false, true, true, false}, .time = {0.0, .1, .2, .3}, .steps = {0, 1, 2, 3}},
                    (EquationIntervalParameters){.equation = "step >= 2", .expectedValues = {false, false, true, true}, .time = {0.0, .1, .2, .3}, .steps = {0, 1, 2, 3}},
                    (EquationIntervalParameters){.equation = "(step >= 1) && (step < 3)", .expectedValues = {false, true, true, false}, .time = {0.0, .1, .2, .3}, .steps = {0, 1, 2, 3}}),
    [](const testing::TestParamInfo<EquationIntervalParameters>& info) { return "equation_" + std::to_string(info.index); });

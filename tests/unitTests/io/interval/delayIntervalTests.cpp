#include <vector>
#include "gtest/gtest.h"
#include "io/interval/delayInterval.hpp"
#include "mockInterval.hpp"

struct DelayIntervalParameters {
    // test param
    std::string name;

    // inputs
    double minimumTime;
    int minimumStep;
    std::function<void(ablateTesting::io::interval::MockInterval&)> setupMockInterval;

    // expected values and inputs
    std::vector<bool> expectedValues;
    std::vector<PetscReal> time;
    std::vector<PetscInt> steps;
};

class DelayIntervalTestFixture : public ::testing::TestWithParam<DelayIntervalParameters> {};

TEST_P(DelayIntervalTestFixture, ShouldProvideCorrectValuesAtCheck) {
    // arrange
    // create and set up the mock interval
    auto mockInterval = std::make_shared<ablateTesting::io::interval::MockInterval>();
    GetParam().setupMockInterval(*mockInterval);

    auto interval = std::make_shared<ablate::io::interval::DelayInterval>(mockInterval, GetParam().minimumTime, GetParam().minimumStep);

    // act/assert
    for (std::size_t i = 0; i < GetParam().expectedValues.size(); i++) {
        auto time = GetParam().time[i];
        auto step = GetParam().steps[i];

        ASSERT_EQ(GetParam().expectedValues[i], interval->Check(MPI_COMM_SELF, step, time));
    }
}

INSTANTIATE_TEST_SUITE_P(DelayIntervalTests, DelayIntervalTestFixture,
                         testing::Values((DelayIntervalParameters){.name = "default_parameters",
                                                                   .minimumTime = 0,
                                                                   .minimumStep = 0,
                                                                   .setupMockInterval =
                                                                       [](ablateTesting::io::interval::MockInterval& mock) {
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 0, .1)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 1, .2)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 2, .3)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 3, .4)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
                                                                       },
                                                                   .expectedValues = {true, false, true, false},
                                                                   .time = {.1, .2, .3, .4},
                                                                   .steps = {0, 1, 2, 3}},
                                         (DelayIntervalParameters){.name = "delayed_step",
                                                                   .minimumTime = 0,
                                                                   .minimumStep = 2,
                                                                   .setupMockInterval =
                                                                       [](ablateTesting::io::interval::MockInterval& mock) {
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 2, .3)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 3, .4)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 4, .5)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                       },
                                                                   .expectedValues = {false, false, true, false, true},
                                                                   .time = {.1, .2, .3, .4, .5},
                                                                   .steps = {0, 1, 2, 3, 4}},
                                         (DelayIntervalParameters){.name = "delayed_time",
                                                                   .minimumTime = .3,
                                                                   .minimumStep = 0,
                                                                   .setupMockInterval =
                                                                       [](ablateTesting::io::interval::MockInterval& mock) {
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 2, .3)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 3, .4)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 4, .5)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                       },
                                                                   .expectedValues = {false, false, true, false, true},
                                                                   .time = {.1, .2, .3, .4, .5},
                                                                   .steps = {0, 1, 2, 3, 4}},
                                         (DelayIntervalParameters){.name = "delayed_time_and_steps",
                                                                   .minimumTime = .3,
                                                                   .minimumStep = 3,
                                                                   .setupMockInterval =
                                                                       [](ablateTesting::io::interval::MockInterval& mock) {
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 3, .4)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 4, .5)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                       },
                                                                   .expectedValues = {false, false, false, true, true},
                                                                   .time = {.1, .2, .3, .4, .5},
                                                                   .steps = {0, 1, 2, 3, 4}},
                                         (DelayIntervalParameters){.name = "delayed_steps_and_time",
                                                                   .minimumTime = .3,
                                                                   .minimumStep = 1,
                                                                   .setupMockInterval =
                                                                       [](ablateTesting::io::interval::MockInterval& mock) {
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 2, .3)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 3, .4)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(false));
                                                                           EXPECT_CALL(mock, Check(MPI_COMM_SELF, 4, .5)).Times(::testing::Exactly(1)).WillOnce(::testing::Return(true));
                                                                       },
                                                                   .expectedValues = {false, false, true, false, true},
                                                                   .time = {.1, .2, .3, .4, .5},
                                                                   .steps = {0, 1, 2, 3, 4}}),
                         [](const testing::TestParamInfo<DelayIntervalParameters>& info) { return info.param.name; });

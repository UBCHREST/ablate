#include "gtest/gtest.h"
#include "monitors/probes.hpp"
#include "temporaryPath.hpp"

class ProbeRecorderFixture : public ::testing::TestWithParam<int> {};

TEST_P(ProbeRecorderFixture, ShouldSaveAndRestart){
    // arrange
    testingResources::TemporaryPath outputPath;
    {
        ablate::monitors::Probes::ProbeRecorder recorder(GetParam(), std::vector<std::string>{"a", "b", "c"}, outputPath.GetPath());

        // act
        recorder.AdvanceTime(0.0);
        recorder.SetValue(0, 1.0);
        recorder.SetValue(1, 2.0);
        recorder.SetValue(2, 3.0);

        recorder.AdvanceTime(0.1);
        recorder.SetValue(0, 4.0);
        recorder.SetValue(1, 5.0);
        recorder.SetValue(2, 6.0);

        recorder.AdvanceTime(0.3);
        recorder.SetValue(0, 7.0);
        recorder.SetValue(1, 8.0);
        recorder.SetValue(2, 9.0);

        recorder.AdvanceTime(0.4);
        recorder.SetValue(1, 11.0);
        recorder.SetValue(2, 12.0);
        recorder.SetValue(0, 10.0);

        recorder.AdvanceTime(0.5);
        recorder.SetValue(0, 13.0);
        recorder.SetValue(1, 14.0);
        recorder.SetValue(2, 15.0);
    }

    {// simulate a restart
        ablate::monitors::Probes::ProbeRecorder recorder(GetParam(), std::vector<std::string>{"a", "b", "c"}, outputPath.GetPath());

        // act
        recorder.AdvanceTime(0.4);
        recorder.SetValue(1, 11.0);
        recorder.SetValue(2, 12.0);
        recorder.SetValue(0, 10.0);

        recorder.AdvanceTime(0.5);
        recorder.SetValue(0, 13.0);
        recorder.SetValue(1, 14.0);
        recorder.SetValue(2, 15.0);

        recorder.AdvanceTime(0.6);
        recorder.SetValue(0, 16.0);
        recorder.SetValue(1, 17.0);
        recorder.SetValue(2, 18.0);

        recorder.AdvanceTime(0.7);
        recorder.SetValue(0, 19.0);
        recorder.SetValue(1, 20.0);
        recorder.SetValue(2, 21.0);
    }

    // assert
    const char *expected =
        "time, a, b, c\n"
        "0.0, 1.0, 2.0, 3.0\n"
        "0.1, 4.0, 5.0, 6.0\n"
        "0.3, 7.0, 8.0, 9.0\n"
        "0.4, 10.0, 11.0, 12.0\n"
        "0.5, 13.0, 14.0, 15.0\n"
        "0.6, 16.0, 17.0, 18.0\n"
        "0.7, 19.0, 20.0, 21.0\n";

    ASSERT_EQ(expected, outputPath.ReadFile().c_str());
}

INSTANTIATE_TEST_SUITE_P(ProbeTests, ProbeRecorderFixture, testing::Values(0, 1, 3, 4, 5, 6));

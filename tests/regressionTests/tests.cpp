#include <filesystem>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"
#include "runners/runners.hpp"

INSTANTIATE_TEST_SUITE_P(
    RegressionRateExamples, RegressionTestsSpecifier,
    testing::Values((MpiTestParameter){
        .testName = "inputs/exampleRegressionTest/exampleRegressionTest.yaml", .nproc = 1, .expectedOutputFile = "outputs/exampleRegressionTest/expectedOutput.txt", .arguments = ""}),

    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(
    MilestoneExamples, RegressionTestsSpecifier,
    testing::Values((MpiTestParameter){.testName = "inputs/slabBurner2D/slabBurner2D.yaml", .nproc = 2, .expectedOutputFile = "outputs/slabBurner2D/expectedOutput.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/slabBurner3D/slabBurner3D.yaml", .nproc = 4, .expectedOutputFile = "outputs/slabBurner3D/expectedOutput.txt", .arguments = ""}),

    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

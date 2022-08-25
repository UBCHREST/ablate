#include <filesystem>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"
#include "runners/runners.hpp"

INSTANTIATE_TEST_SUITE_P(
    RegressionRateExamples, RegressionTestsSpecifier,
    testing::Values((MpiTestParameter){
        .testName = "inputs/exampleRegressionTest/exampleRegressionTest.yaml", .nproc = 1, .expectedOutputFile = "outputs/exampleRegressionTest/expectedOutput.txt", .arguments = ""}),

    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

#include <filesystem>
#include "gtest/gtest.h"
#include "mpiTestFixture.hpp"
#include "runners/runners.hpp"

INSTANTIATE_TEST_SUITE_P(RegressionRateExamples, RegressionTestsSpecifier,
                         testing::Values(MpiTestParameter("inputs/exampleRegressionTest/exampleRegressionTest.yaml", 1, "", "outputs/exampleRegressionTest/expectedOutput.txt")),

                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(MilestoneExamples, RegressionTestsSpecifier,
                         testing::Values(MpiTestParameter("inputs/slabBurner2D/slabBurner2D.yaml", 2, "", "outputs/slabBurner2D/expectedOutput.txt"),
                                         MpiTestParameter("inputs/slabBurner3D/slabBurner3D.yaml", 4, "", "outputs/slabBurner3D/expectedOutput.txt")),

                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

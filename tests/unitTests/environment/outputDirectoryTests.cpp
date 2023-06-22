#include "environment//outputDirectory.hpp"
#include "environment/gitHub.hpp"
#include "gtest/gtest.h"
#include "testRunEnvironment.hpp"

TEST(OutputDirectoryTests, ShouldReportRelativePath) {
    // ARRANGE
    // setup the mock parameters
    auto tempDir = std::filesystem::temp_directory_path();
    testingResources::TestRunEnvironment testRunEnvironment(tempDir);

    std::string fileName = "path/to/file.txt";

    // ACT
    ablate::environment::OutputDirectory outputDirectory(fileName);

    // ASSERT
    ASSERT_EQ(tempDir / fileName, outputDirectory.Locate({}));
}
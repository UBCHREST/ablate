#include <PetscTestFixture.hpp>
#include <fstream>
#include <memory>
#include "environment/download.hpp"
#include "gtest/gtest.h"
#include "testRunEnvironment.hpp"

#define REMOTE_URL "https://raw.githubusercontent.com/UBCHREST/ablate/main/tests/ablateLibrary/inputs/eos/thermo30.dat"

class DownloadTestsFixture : public testingResources::PetscTestFixture {};

TEST_F(DownloadTestsFixture, ShouldDownloadFile) {
    // arrange
    ablate::environment::Download fileLocator(REMOTE_URL);

    // act
    auto computedFilePath = fileLocator.Locate();

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));

    // cleanup
    fs::remove(computedFilePath);
}

TEST_F(DownloadTestsFixture, ShouldDownloadAndRelocateFile) {
    // arrange
    fs::path outputDir = fs::temp_directory_path() / "outputDirTemp";
    std::filesystem::create_directories(outputDir);

    // setup the run env
    testingResources::TestRunEnvironment testRunEnvironment(outputDir);

    ablate::environment::Download fileLocator(REMOTE_URL);

    // act
    auto computedFilePath = fileLocator.Locate();

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(outputDir, computedFilePath.parent_path());

    // cleanup
    fs::remove(computedFilePath);
    fs::remove_all(outputDir);
}
#include <PetscTestFixture.hpp>
#include <fstream>
#include <memory>
#include "environment/gitHub.hpp"
#include "gtest/gtest.h"
#include "testRunEnvironment.hpp"

class GitHubTestsFixture : public testingResources::PetscTestFixture {};

TEST_F(GitHubTestsFixture, ShouldDownloadDirectory) {
    // check for test
    if (const char* gitHubToken = std::getenv("GITHUB_TOKEN")) {
        // arrange
        ablate::environment::GitHub fileLocator("ubchrest/ablate", "tests/unitTests/inputs", gitHubToken);

        // act
        auto computedFilePath = fileLocator.Locate();

        // assert
        ASSERT_TRUE(std::filesystem::exists(computedFilePath));
        ASSERT_FALSE(std::filesystem::is_empty(computedFilePath));

        // cleanup
        fs::remove_all(computedFilePath);
    } else {
        SUCCEED() << "Test is only applicable when GITHUB_TOKEN env is defined";
    }
}

TEST_F(GitHubTestsFixture, ShouldDownloadAndRelocateDirectory) {
    // check for test
    if (const char* gitHubToken = std::getenv("GITHUB_TOKEN")) {
        // arrange
        fs::path outputDir = fs::temp_directory_path() / "outputDirTemp";
        std::filesystem::create_directories(outputDir);

        // setup the run env
        testingResources::TestRunEnvironment testRunEnvironment(outputDir, "");

        ablate::environment::GitHub fileLocator("ubchrest/ablate", "tests/unitTests/inputs", gitHubToken);

        // act
        auto computedFilePath = fileLocator.Locate();

        // assert
        ASSERT_TRUE(std::filesystem::exists(computedFilePath));
        ASSERT_EQ(outputDir, computedFilePath.parent_path());
        ASSERT_FALSE(std::filesystem::is_empty(computedFilePath));

        // cleanup
        fs::remove_all(computedFilePath);
        fs::remove_all(outputDir);
    } else {
        SUCCEED() << "Test is only applicable when GITHUB_TOKEN env is defined";
    }
}

TEST_F(GitHubTestsFixture, ShouldDownloadFile) {
    // check for test
    if (const char* gitHubToken = std::getenv("GITHUB_TOKEN")) {
        // arrange
        ablate::environment::GitHub fileLocator("ubchrest/ablate", "tests/unitTests/inputs/eos/gri30.yaml", gitHubToken);

        // act
        auto computedFilePath = fileLocator.Locate();

        // assert
        ASSERT_EQ(std::string("gri30.yaml"), computedFilePath.filename());
        ASSERT_TRUE(std::filesystem::exists(computedFilePath));
        ASSERT_FALSE(std::filesystem::is_empty(computedFilePath));

        // cleanup
        fs::remove_all(computedFilePath);
    } else {
        SUCCEED() << "Test is only applicable when GITHUB_TOKEN env is defined";
    }
}

TEST_F(GitHubTestsFixture, ShouldDownloadAndRelocateFile) {
    // check for test
    if (const char* gitHubToken = std::getenv("GITHUB_TOKEN")) {
        // arrange
        fs::path outputDir = fs::temp_directory_path() / "outputDirTemp";
        std::filesystem::create_directories(outputDir);

        // setup the run env
        testingResources::TestRunEnvironment testRunEnvironment(outputDir);

        ablate::environment::GitHub fileLocator("ubchrest/ablate", "tests/unitTests/inputs/eos/gri30.yaml", gitHubToken);

        // act
        auto computedFilePath = fileLocator.Locate();

        // assert
        ASSERT_EQ(std::string("gri30.yaml"), computedFilePath.filename());
        ASSERT_TRUE(std::filesystem::exists(computedFilePath));
        ASSERT_EQ(outputDir, computedFilePath.parent_path());
        ASSERT_FALSE(std::filesystem::is_empty(computedFilePath));

        // cleanup
        fs::remove_all(computedFilePath);
        fs::remove_all(outputDir);
    } else {
        SUCCEED() << "Test is only applicable when GITHUB_TOKEN env is defined";
    }
}
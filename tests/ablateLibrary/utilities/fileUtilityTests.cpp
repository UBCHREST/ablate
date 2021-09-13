#include <PetscTestFixture.hpp>
#include <fstream>
#include <memory>
#include <sstream>
#include "gtest/gtest.h"
#include "utilities/fileUtility.hpp"

#define REMOTE_URL "https://raw.githubusercontent.com/UBCHREST/ablate/main/tests/ablateLibrary/inputs/eos/thermo30.dat"

TEST(FileUtilityTets, ShouldLocateFileInSearchPaths) {
    // arrange
    std::filesystem::path directory = std::filesystem::temp_directory_path() / "tmpDir";
    std::filesystem::create_directories(directory);
    std::filesystem::path tmpFile = directory / "tempFile.txt";
    {
        std::ofstream ofs(tmpFile);
        ofs << " tempFile" << std::endl;
        ofs.close();
    }

    // act
    auto computedFilePath = ablate::utilities::FileUtility::LocateFile("tempFile.txt", MPI_COMM_SELF, {directory});

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(computedFilePath, std::filesystem::canonical(tmpFile));

    // cleanup
    fs::remove(tmpFile);
    fs::remove_all(directory);
}

TEST(FileUtilityTets, ShouldLocateRelativeFileInSearchPathsAndReturnCanonicalPath) {
    // arrange
    std::filesystem::path directory = std::filesystem::temp_directory_path() / "tmpDir";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    std::filesystem::path otherDirectory = std::filesystem::temp_directory_path() / "otherDir";
    std::filesystem::remove_all(otherDirectory);
    std::filesystem::create_directories(otherDirectory);

    std::filesystem::path tmpFile = otherDirectory / "tempFile.txt";
    {
        std::ofstream ofs(tmpFile);
        ofs << " tempFile" << std::endl;
        ofs.close();
    }

    // act
    auto computedFilePath = ablate::utilities::FileUtility::LocateFile("../otherDir/tempFile.txt", MPI_COMM_SELF, {directory});

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(computedFilePath, std::filesystem::canonical(tmpFile));

    // cleanup
    fs::remove(tmpFile);
    fs::remove_all(directory);
    std::filesystem::remove_all(otherDirectory);
}

class FileUtilityTestsTestFixture : public testingResources::PetscTestFixture {};

TEST_F(FileUtilityTestsTestFixture, ShouldDownloadFile) {
    // arrange
    // act
    auto computedFilePath = ablate::utilities::FileUtility::LocateFile(REMOTE_URL, MPI_COMM_SELF);

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));

    // cleanup
    fs::remove(computedFilePath);
}

TEST_F(FileUtilityTestsTestFixture, ShouldDownloadAndRelocateFile) {
    // arrange
    fs::path outputDir = fs::temp_directory_path() / "outputDirTemp";
    std::filesystem::create_directories(outputDir);

    // act
    auto computedFilePath = ablate::utilities::FileUtility::LocateFile(REMOTE_URL, MPI_COMM_SELF, {}, outputDir);

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(outputDir, computedFilePath.parent_path());

    // cleanup
    fs::remove(computedFilePath);
    fs::remove_all(outputDir);
}

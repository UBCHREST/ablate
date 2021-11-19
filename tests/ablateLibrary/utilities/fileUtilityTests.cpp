#include <PetscTestFixture.hpp>
#include <fstream>
#include <memory>
#include <sstream>
#include "gtest/gtest.h"
#include "utilities/fileUtility.hpp"

#define REMOTE_URL "https://raw.githubusercontent.com/UBCHREST/ablate/main/tests/ablateLibrary/inputs/eos/thermo30.dat"

TEST(FileUtilityTests, ShouldLocateFileInSearchPaths) {
    // arrange
    std::filesystem::path directory = std::filesystem::temp_directory_path() / "tmpDir";
    std::filesystem::create_directories(directory);
    std::filesystem::path tmpFile = directory / "tempFile.txt";
    {
        std::ofstream ofs(tmpFile);
        ofs << " tempFile" << std::endl;
        ofs.close();
    }

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {directory});

    // act
    auto computedFilePath = fileLocator.Locate("tempFile.txt");

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(computedFilePath, std::filesystem::canonical(tmpFile));

    // cleanup
    fs::remove(tmpFile);
    fs::remove_all(directory);
}

TEST(FileUtilityTests, FunctionShouldLocateFileInSearchPaths) {
    // arrange
    std::filesystem::path directory = std::filesystem::temp_directory_path() / "tmpDir";
    std::filesystem::create_directories(directory);
    std::filesystem::path tmpFile = directory / "tempFile.txt";
    {
        std::ofstream ofs(tmpFile);
        ofs << " tempFile" << std::endl;
        ofs.close();
    }

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {directory});

    // act
    auto computedFilePath = fileLocator.GetLocateFileFunction()("tempFile.txt");

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(computedFilePath, std::filesystem::canonical(tmpFile));

    // cleanup
    fs::remove(tmpFile);
    fs::remove_all(directory);
}

TEST(FileUtilityTests, StaticShouldLocateFileInSearchPaths) {
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

TEST(FileUtilityTests, ShouldLocateRelativeFileInSearchPathsAndReturnCanonicalPath) {
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

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {directory});

    // act
    auto computedFilePath = fileLocator.Locate("../otherDir/tempFile.txt");

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(computedFilePath, std::filesystem::canonical(tmpFile));

    // cleanup
    fs::remove(tmpFile);
    fs::remove_all(directory);
    std::filesystem::remove_all(otherDirectory);
}

TEST(FileUtilityTests, FunctionShouldLocateRelativeFileInSearchPathsAndReturnCanonicalPath) {
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

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {directory});

    // act
    auto computedFilePath = fileLocator.GetLocateFileFunction()("../otherDir/tempFile.txt");

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(computedFilePath, std::filesystem::canonical(tmpFile));

    // cleanup
    fs::remove(tmpFile);
    fs::remove_all(directory);
    std::filesystem::remove_all(otherDirectory);
}

TEST(FileUtilityTests, StaticShouldLocateRelativeFileInSearchPathsAndReturnCanonicalPath) {
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
    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF);

    // act
    auto computedFilePath = fileLocator.Locate(REMOTE_URL);

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));

    // cleanup
    fs::remove(computedFilePath);
}

TEST_F(FileUtilityTestsTestFixture, StaticShouldDownloadFile) {
    // arrange
    // act
    auto computedFilePath = ablate::utilities::FileUtility::LocateFile(REMOTE_URL, MPI_COMM_SELF);

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));

    // cleanup
    fs::remove(computedFilePath);
}

TEST_F(FileUtilityTestsTestFixture, FunctionShouldDownloadFile) {
    // arrange
    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF);

    // act
    auto computedFilePath = fileLocator.GetLocateFileFunction()(REMOTE_URL);

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));

    // cleanup
    fs::remove(computedFilePath);
}

TEST_F(FileUtilityTestsTestFixture, ShouldDownloadAndRelocateFile) {
    // arrange
    fs::path outputDir = fs::temp_directory_path() / "outputDirTemp";
    std::filesystem::create_directories(outputDir);

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {}, outputDir);

    // act
    auto computedFilePath = fileLocator.Locate(REMOTE_URL);

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(outputDir, computedFilePath.parent_path());

    // cleanup
    fs::remove(computedFilePath);
    fs::remove_all(outputDir);
}

TEST_F(FileUtilityTestsTestFixture, FunctionShouldDownloadAndRelocateFile) {
    // arrange
    fs::path outputDir = fs::temp_directory_path() / "outputDirTemp";
    std::filesystem::create_directories(outputDir);

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {}, outputDir);

    // act
    auto computedFilePath = fileLocator.GetLocateFileFunction()(REMOTE_URL);

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(outputDir, computedFilePath.parent_path());

    // cleanup
    fs::remove(computedFilePath);
    fs::remove_all(outputDir);
}

TEST_F(FileUtilityTestsTestFixture, StaticShouldDownloadAndRelocateFile) {
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

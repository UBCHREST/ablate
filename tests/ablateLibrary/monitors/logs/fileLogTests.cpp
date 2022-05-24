#include <petsc.h>
#include <MpiTestParamFixture.hpp>
#include <cmath>
#include <fstream>
#include <memory>
#include "MpiTestFixture.hpp"
#include "environment/runEnvironment.hpp"
#include "PetscTestErrorChecker.hpp"
#include "gtest/gtest.h"
#include "monitors/logs/fileLog.hpp"
#include "parameters/mapParameters.hpp"
#include "utilities/petscUtilities.hpp"
#include "testRunEnvironment.hpp"

using namespace ablate;

class FileLogTestFixture : public testingResources::MpiTestParamFixture {};

TEST_P(FileLogTestFixture, ShouldPrintToFile) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            ablate::utilities::PetscUtilities::Initialize(argc, argv);

            // Create the fileLog
            auto logPath = MakeTemporaryPath("logFile.txt");
            monitors::logs::FileLog log(logPath);
            log.Initialize(PETSC_COMM_WORLD);

            // Get the current rank
            PetscMPIInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            log.Print("Log Out Log\n");
            log.Printf("rank: %d\n", rank);
        }
        ablate::environment::RunEnvironment::Get().CleanUp();
        exit(0);
    EndWithMPI

    // Load the file
    std::ifstream logFile(std::filesystem::temp_directory_path() / "logFile.txt");
    std::stringstream buffer;
    buffer << logFile.rdbuf();

    ASSERT_EQ(buffer.str(), "Log Out Log\nrank: 0\n");
}

TEST_P(FileLogTestFixture, ShouldPrintToFileInOutputDirectory) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            ablate::utilities::PetscUtilities::Initialize(argc, argv);

            // Set the global environment
            auto tempDir = MakeTemporaryPath("nameOfTestDir", PETSC_COMM_WORLD);
            testingResources::TestRunEnvironment testRunEnvironment(tempDir, "");

            // Create the fileLog
            monitors::logs::FileLog log("logFile.txt");
            log.Initialize(PETSC_COMM_WORLD);

            // Get the current rank
            PetscMPIInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            log.Print("Log Out Log\n");
            log.Printf("rank: %d\n", rank);
        }
        ablate::environment::RunEnvironment::Get().CleanUp();
        exit(0);
    EndWithMPI

    // Load the file
    auto logFilePath = std::filesystem::temp_directory_path() / "nameOfTestDir" / "logFile.txt";
    std::ifstream logFile(logFilePath);
    std::stringstream buffer;
    buffer << logFile.rdbuf();

    ASSERT_EQ(buffer.str(), "Log Out Log\nrank: 0\n");
}

TEST_P(FileLogTestFixture, ShouldAppendToFileInOutputDirectory) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            ablate::utilities::PetscUtilities::Initialize(argc, argv);

            // Set the global environment
            auto tempDir = MakeTemporaryPath("nameOfTestDir", PETSC_COMM_WORLD);
            testingResources::TestRunEnvironment testRunEnvironment(tempDir);

            {
                // Create the fileLog
                monitors::logs::FileLog log("logFile.txt");
                log.Initialize(PETSC_COMM_WORLD);

                // Get the current rank
                PetscMPIInt rank;
                MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

                log.Print("Log Out Log\n");
                log.Printf("rank: %d\n", rank);
            }
            {
                // Create the fileLog
                monitors::logs::FileLog log("logFile.txt");
                log.Initialize(PETSC_COMM_WORLD);

                // Get the current rank
                PetscMPIInt rank;
                MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

                log.Print("Log Out 2\n");
                log.Printf("rank: %d\n", rank);
            }
        }
        ablate::environment::RunEnvironment::Get().CleanUp();
        exit(0);
    EndWithMPI

    // Load the file
    auto logFilePath = std::filesystem::temp_directory_path() / "nameOfTestDir" / "logFile.txt";
    std::ifstream logFile(logFilePath);
    std::stringstream buffer;
    buffer << logFile.rdbuf();

    ASSERT_EQ(buffer.str(), "Log Out Log\nrank: 0\nLog Out 2\nrank: 0\n");
}

INSTANTIATE_TEST_SUITE_P(LogTests, FileLogTestFixture, testing::Values((MpiTestParameter){.testName = "logFile 1 proc", .nproc = 1, .arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter> &info) { return info.param.getTestName(); });

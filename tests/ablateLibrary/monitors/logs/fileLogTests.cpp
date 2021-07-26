#include <petsc.h>
#include <MpiTestParamFixture.hpp>
#include <cmath>
#include "environment/runEnvironment.hpp"
#include <fstream>
#include <memory>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "gtest/gtest.h"
#include "monitors/logs/fileLog.hpp"
#include "parameters/mapParameters.hpp"

using namespace ablate;

class FileLogTestFigure : public testingResources::MpiTestParamFixture {};

TEST_P(FileLogTestFigure, ShouldPrintToFile) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            PetscInitialize(argc, argv, NULL, NULL) >> testErrorChecker;

            // Create the fileLog
            monitors::logs::FileLog log("logFile.txt");
            log.Initialize(PETSC_COMM_WORLD);

            // Get the current rank
            PetscInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            log.Print("Log Out Log\n");
            log.Printf("rank: %d\n", rank);
        }
        exit(PetscFinalize());
    EndWithMPI

    // Load the file
    std::ifstream logFile("logFile.txt");
    std::stringstream buffer;
    buffer << logFile.rdbuf();

    ASSERT_EQ(buffer.str(), "Log Out Log\nrank: 0\n");
}

TEST_P(FileLogTestFigure, ShouldPrintToFileInOutputDirectory) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            PetscInitialize(argc, argv, NULL, NULL) >> testErrorChecker;

            // Set the global environment
            auto tempDir = std::filesystem::temp_directory_path() / "nameOfTestDir";
            parameters::MapParameters parameters ({{"outputDirectory", tempDir}, {"title", ""}, {"tagDirectory", "false"}});
            environment::RunEnvironment::Setup(parameters);

            // Create the fileLog
            monitors::logs::FileLog log("logFile.txt");
            log.Initialize(PETSC_COMM_WORLD);

            // Get the current rank
            PetscInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            log.Print("Log Out Log\n");
            log.Printf("rank: %d\n", rank);
        }
        exit(PetscFinalize());
    EndWithMPI

    // Load the file
    auto logFilePath = std::filesystem::temp_directory_path() / "nameOfTestDir" / "logFile.txt";
    std::ifstream logFile(logFilePath);
    std::stringstream buffer;
    buffer << logFile.rdbuf();

    ASSERT_EQ(buffer.str(), "Log Out Log\nrank: 0\n");
}

INSTANTIATE_TEST_SUITE_P(LogTests, FileLogTestFigure,
                         testing::Values((MpiTestParameter){.testName = "logFile 1 proc", .nproc = 1,.arguments = ""},
                                         (MpiTestParameter){.testName = "logFile 2 proc", .nproc = 2,.arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter> &info) { return info.param.getTestName(); });

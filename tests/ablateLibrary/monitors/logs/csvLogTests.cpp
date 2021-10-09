#include <petsc.h>
#include <MpiTestParamFixture.hpp>
#include <cmath>
#include <fstream>
#include <memory>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "monitors/logs/csvLog.hpp"
#include "parameters/mapParameters.hpp"

using namespace ablate;

class CsvLogTestFixture : public testingResources::MpiTestParamFixture {};

TEST_P(CsvLogTestFixture, ShouldPrintToFile) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            PetscInitialize(argc, argv, NULL, NULL) >> testErrorChecker;

            // Create the fileLog
            auto logPath = MakeTemporaryPath("logFile.csv", PETSC_COMM_WORLD);
            std::shared_ptr<monitors::logs::Log> log = std::make_shared<monitors::logs::CsvLog>(logPath);
            log->Initialize(PETSC_COMM_WORLD);

            // Get the current rank
            PetscMPIInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            // pretend we are going over multiple iterations
            log->Print("Log Out Log\n");
            log->Printf("rank: %d iter %d\n", rank, 0);
            std::vector<double> otherData = {2.2, 3.3, 4.4};
            log->Print("otherData", otherData);

            log->Print("Log Out Log\n");
            log->Printf("rank: %d iter %d\n", rank, 1);
            otherData = {5.5, 6.6, 7.7};
            log->Print("otherData", otherData);
        }
        exit(PetscFinalize());
    EndWithMPI

    // Load the file
    std::ifstream logFile(std::filesystem::temp_directory_path() / "logFile.csv");
    std::stringstream buffer;
    buffer << logFile.rdbuf();

    ASSERT_EQ(buffer.str(), "\n0,0,2.2,3.3,4.4,\n0,1,5.5,6.6,7.7,");
}

TEST_P(CsvLogTestFixture, ShouldPrintToFileInOutputDirectory) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            PetscInitialize(argc, argv, NULL, NULL) >> testErrorChecker;

            // Set the global environment
            auto tempDir = MakeTemporaryPath("nameOfTestDir", PETSC_COMM_WORLD);
            parameters::MapParameters parameters({{"directory", tempDir}, {"title", ""}, {"tagDirectory", "false"}});
            environment::RunEnvironment::Setup(parameters);

            // Create the fileLog
            std::shared_ptr<monitors::logs::Log> log = std::make_shared<monitors::logs::CsvLog>("logFile.csv");
            log->Initialize(PETSC_COMM_WORLD);

            // Get the current rank
            PetscMPIInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            // pretend we are going over multiple iterations
            log->Print("Log Out Log\n");
            log->Printf("rank: %d iter %d\n", rank, 0);
            std::vector<double> otherData = {2.2, 3.3, 4.4};
            log->Print("otherData", otherData);

            log->Print("Log Out Log\n");
            log->Printf("rank: %d iter %d\n", rank, 1);
            otherData = {5.5, 6.6, 7.7};
            log->Print("otherData", otherData);
        }
        exit(PetscFinalize());
    EndWithMPI

    // Load the file
    auto logFilePath = std::filesystem::temp_directory_path() / "nameOfTestDir" / "logFile.csv";
    std::ifstream logFile(logFilePath);
    std::stringstream buffer;
    buffer << logFile.rdbuf();

    ASSERT_EQ(buffer.str(), "\n0,0,2.2,3.3,4.4,\n0,1,5.5,6.6,7.7,");
}

TEST_P(CsvLogTestFixture, ShouldAppendToFileInOutputDirectory) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            PetscInitialize(argc, argv, NULL, NULL) >> testErrorChecker;

            // Set the global environment
            auto tempDir = MakeTemporaryPath("nameOfTestDir", PETSC_COMM_WORLD);
            parameters::MapParameters parameters({{"directory", tempDir}, {"title", ""}, {"tagDirectory", "false"}});
            environment::RunEnvironment::Setup(parameters);

            // Create the fileLog
            {
                std::shared_ptr<monitors::logs::Log> log = std::make_shared<monitors::logs::CsvLog>("logFile.csv");
                log->Initialize(PETSC_COMM_WORLD);

                // Get the current rank
                PetscMPIInt rank;
                MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

                // pretend we are going over multiple iterations
                log->Print("Log Out Log\n");
                log->Printf("rank: %d iter %d\n", rank, 0);
                std::vector<double> otherData = {2.2, 3.3, 4.4};
                log->Print("otherData", otherData);

                log->Print("Log Out Log\n");
                log->Printf("rank: %d iter %d\n", rank, 1);
                otherData = {5.5, 6.6, 7.7};
                log->Print("otherData", otherData);
            }

            {  // Reopen and append
                std::shared_ptr<monitors::logs::Log> log = std::make_shared<monitors::logs::CsvLog>("logFile.csv");
                log->Initialize(PETSC_COMM_WORLD);

                // Get the current rank
                PetscMPIInt rank;
                MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

                // pretend we are going over multiple iterations
                log->Print("Log Out Log\n");
                log->Printf("rank: %d iter %d\n", rank, 0);
                std::vector<double> otherData = {8.8, 9.9, 10.10};
                log->Print("otherData", otherData);

                log->Print("Log Out Log\n");
                log->Printf("rank: %d iter %d\n", rank, 1);
                otherData = {11.11, 12.12, 13.13};
                log->Print("otherData", otherData);
            }
        }
        exit(PetscFinalize());
    EndWithMPI

    // Load the file
    auto logFilePath = std::filesystem::temp_directory_path() / "nameOfTestDir" / "logFile.csv";
    std::ifstream logFile(logFilePath);
    std::stringstream buffer;
    buffer << logFile.rdbuf();

    ASSERT_EQ(buffer.str(), "\n0,0,2.2,3.3,4.4,\n0,1,5.5,6.6,7.7,\n0,0,8.8,9.9,10.1,\n0,1,11.11,12.12,13.13,");
}

INSTANTIATE_TEST_SUITE_P(LogTests, CsvLogTestFixture,
                         testing::Values((MpiTestParameter){.testName = "csvFile 1 proc", .nproc = 1, .arguments = ""}, (MpiTestParameter){.testName = "csvFile 2 proc", .nproc = 2, .arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter> &info) { return info.param.getTestName(); });

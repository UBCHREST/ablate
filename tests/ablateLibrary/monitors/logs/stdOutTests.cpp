#include <petsc.h>
#include <MpiTestParamFixture.hpp>
#include <cmath>
#include <memory>
#include "MpiTestFixture.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "monitors/logs/stdOut.hpp"
#include "utilities/petscUtilities.hpp"

using namespace ablate;

class StdOutLogTestFixture : public testingResources::MpiTestParamFixture {};

TEST_P(StdOutLogTestFixture, ShouldPrintToStdOut) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            ablate::utilities::PetscUtilities::Initialize(argc, argv);

            // Create the stdOut
            monitors::logs::StdOut log;
            log.Initialize(PETSC_COMM_WORLD);

            // Get the current rank
            PetscMPIInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            log.Print("Standard Out Log\n");
            log.Printf("rank: %d\n", rank);
        }
        ablate::environment::RunEnvironment::Get().CleanUp();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(LogTests, StdOutLogTestFixture,
                         testing::Values((MpiTestParameter){.testName = "std out 1 proc", .nproc = 1, .expectedOutputFile = "outputs/monitors/logs/stdOutLogFile", .arguments = ""},
                                         (MpiTestParameter){.testName = "std out 2 proc", .nproc = 2, .expectedOutputFile = "outputs/monitors/logs/stdOutLogFile", .arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter> &info) { return info.param.getTestName(); });

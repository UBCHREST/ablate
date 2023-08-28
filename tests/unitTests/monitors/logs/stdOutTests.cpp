#include <petsc.h>
#include <cmath>
#include <memory>
#include <mpiTestParamFixture.hpp>
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "monitors/logs/stdOut.hpp"
#include "mpiTestFixture.hpp"
#include "utilities/petscUtilities.hpp"

using namespace ablate;

class StdOutLogTestFixture : public testingResources::MpiTestParamFixture {};

TEST_P(StdOutLogTestFixture, ShouldPrintToStdOut) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();

            // Create the stdOut
            monitors::logs::StdOut log;
            log.Initialize(PETSC_COMM_WORLD);

            // Get the current rank
            PetscMPIInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            log.Print("Standard Out Log\n");
            log.Printf("rank: %d\n", rank);

            // Should also print from a stream
            auto& stream = log.GetStream();
            stream << "stream: " << rank << std::endl;
        }
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(LogTests, StdOutLogTestFixture,
                         testing::Values(testingResources::MpiTestParameter("std out 1 proc", 1, "", "outputs/monitors/logs/stdOutLogFile"),
                                         testingResources::MpiTestParameter("std out 2 proc", 2, "", "outputs/monitors/logs/stdOutLogFile")),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

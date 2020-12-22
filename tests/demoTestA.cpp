#include "gtest/gtest.h"
#include "MpiTestFixture.h"
#include <mpi.h>

class DemoTestA : public MpiTestFixture{
public:
    char green = 4;
};

TEST_P(DemoTestA, DemoTesta1) {
    StartWithMPI
        // Initialize the MPI environment
        MPI_Init(argc, argv);

        // Get the number of processes
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Get the rank of the process
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Print off a hello world message
        for(int r =0; r < world_size; r++) {
            if(r == world_rank) {
                printf("Hello world from rank %d out of %d processors\n",
                       world_rank, world_size);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Finalize the MPI environment.
        MPI_Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(TestOne, DemoTestA, testing::Values(
        (MpiTestParameter){.nproc = 3, .expectedOutputFile="outputs/output.test.112", .arguments="one two three"},
        (MpiTestParameter){.nproc = 1, .expectedOutputFile="outputs/output.test.112", .arguments=""}
));
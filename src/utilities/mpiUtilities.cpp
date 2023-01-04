#include "mpiUtilities.hpp"

void ablate::utilities::MpiUtilities::RoundRobin(MPI_Comm comm, std::function<void(int)> function) {
    int size = 1;
    int rank = 0;

    int mpiInitialized;
    MPI_Initialized(&mpiInitialized) >> utilities::MpiUtilities::checkError;
    if (mpiInitialized) {
        MPI_Comm_rank(comm, &rank) >> utilities::MpiUtilities::checkError;
        MPI_Comm_size(comm, &size) >> utilities::MpiUtilities::checkError;
    }

    // call each function one at a time
    for (int r = 0; r < size; r++) {
        if (r == rank) {
            function(rank);
        }
        if (mpiInitialized) {
            MPI_Barrier(PETSC_COMM_WORLD);
        }
    }
}

void ablate::utilities::MpiUtilities::Once(MPI_Comm comm, std::function<void()> function, int root) {
    int rank = 0;

    int mpiInitialized;
    MPI_Initialized(&mpiInitialized) >> utilities::MpiUtilities::checkError;
    if (mpiInitialized) {
        MPI_Comm_rank(comm, &rank) >> utilities::MpiUtilities::checkError;
    }

    // call each function one at a time
    if (rank == root) {
        function();
    }

    if (mpiInitialized) {
        MPI_Barrier(comm) >> utilities::MpiUtilities::checkError;
    }
}
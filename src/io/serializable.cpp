#include "serializable.hpp"
#include "utilities/mpiError.hpp"
#include "utilities/petscError.hpp"

void ablate::io::Serializable::SaveKeyValue(PetscViewer viewer, const char *name, PetscScalar value) {
    PetscMPIInt rank;
    MPI_Comm comm = PetscObjectComm((PetscObject)viewer);
    MPI_Comm_rank(comm, &rank) >> checkMpiError;

    // create a very simple vector
    Vec keyValueVec;
    VecCreateMPI(comm, rank == 0 ? 1 : 0, 1, &keyValueVec) >> checkError;
    PetscObjectSetName((PetscObject)keyValueVec, name) >> checkError;
    if (rank == 0) {
        PetscInt globOwnership = 0;
        VecSetValues(keyValueVec, 1, &globOwnership, &value, INSERT_VALUES) >> checkError;
    }
    VecAssemblyBegin(keyValueVec) >> checkError;
    VecAssemblyEnd(keyValueVec) >> checkError;
    VecView(keyValueVec, viewer);
    VecDestroy(&keyValueVec) >> checkError;
}

void ablate::io::Serializable::RestoreKeyValue(PetscViewer viewer, const char *name, PetscScalar &value) {
    int rank;
    MPI_Comm comm = PetscObjectComm((PetscObject)viewer);
    MPI_Comm_rank(comm, &rank) >> checkMpiError;

    // load in the old alpha
    Vec keyValueVec;
    VecCreateMPI(comm, rank == 0 ? 1 : 0, 1, &keyValueVec) >> checkError;
    PetscObjectSetName((PetscObject)keyValueVec, name) >> checkError;
    VecLoad(keyValueVec, viewer) >> checkError;

    // Load in value
    if (rank == 0) {
        PetscInt index[1] = {0};
        VecGetValues(keyValueVec, 1, index, &value) >> checkError;
    }

    // Broadcast everywhere
    MPI_Bcast(&value, 1, MPIU_SCALAR, 0, comm) >> checkMpiError;
    VecDestroy(&keyValueVec) >> checkError;
}

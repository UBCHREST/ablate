#include "serializable.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

void ablate::io::Serializable::SaveKeyValue(PetscViewer viewer, const char *name, PetscScalar value) {
    PetscMPIInt rank;
    MPI_Comm comm = PetscObjectComm((PetscObject)viewer);
    MPI_Comm_rank(comm, &rank) >> utilities::MpiUtilities::checkError;

    // create a very simple vector
    Vec keyValueVec;
    VecCreateMPI(comm, rank == 0 ? 1 : 0, 1, &keyValueVec) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)keyValueVec, name) >> utilities::PetscUtilities::checkError;
    if (rank == 0) {
        PetscInt globOwnership = 0;
        VecSetValues(keyValueVec, 1, &globOwnership, &value, INSERT_VALUES) >> utilities::PetscUtilities::checkError;
    }
    VecAssemblyBegin(keyValueVec) >> utilities::PetscUtilities::checkError;
    VecAssemblyEnd(keyValueVec) >> utilities::PetscUtilities::checkError;
    VecView(keyValueVec, viewer);
    VecDestroy(&keyValueVec) >> utilities::PetscUtilities::checkError;
}

void ablate::io::Serializable::RestoreKeyValue(PetscViewer viewer, const char *name, PetscScalar &value) {
    int rank;
    MPI_Comm comm = PetscObjectComm((PetscObject)viewer);
    MPI_Comm_rank(comm, &rank) >> utilities::MpiUtilities::checkError;

    // load in the old alpha
    Vec keyValueVec;
    VecCreateMPI(comm, rank == 0 ? 1 : 0, 1, &keyValueVec) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)keyValueVec, name) >> utilities::PetscUtilities::checkError;
    VecLoad(keyValueVec, viewer) >> utilities::PetscUtilities::checkError;

    // Load in value
    if (rank == 0) {
        PetscInt index[1] = {0};
        VecGetValues(keyValueVec, 1, index, &value) >> utilities::PetscUtilities::checkError;
    }

    // Broadcast everywhere
    MPI_Bcast(&value, 1, MPIU_SCALAR, 0, comm) >> utilities::MpiUtilities::checkError;
    VecDestroy(&keyValueVec) >> utilities::PetscUtilities::checkError;
}

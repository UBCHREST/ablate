#include "serializable.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

PetscErrorCode ablate::io::Serializable::SaveKeyValue(PetscViewer viewer, const char *name, PetscScalar value) {
    PetscFunctionBeginUser;
    PetscMPIInt rank;
    MPI_Comm comm = PetscObjectComm((PetscObject)viewer);
    PetscCallMPI(MPI_Comm_rank(comm, &rank));

    // create a very simple vector
    Vec keyValueVec;
    PetscCall(VecCreateMPI(comm, rank == 0 ? 1 : 0, 1, &keyValueVec));
    PetscCall(PetscObjectSetName((PetscObject)keyValueVec, name));
    if (rank == 0) {
        PetscInt globOwnership = 0;
        PetscCall(VecSetValues(keyValueVec, 1, &globOwnership, &value, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(keyValueVec));
    PetscCall(VecAssemblyEnd(keyValueVec));
    PetscCall(VecView(keyValueVec, viewer));
    PetscCall(VecDestroy(&keyValueVec));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::io::Serializable::RestoreKeyValue(PetscViewer viewer, const char *name, PetscScalar &value) {
    PetscFunctionBeginUser;
    int rank;
    MPI_Comm comm = PetscObjectComm((PetscObject)viewer);
    PetscCallMPI(MPI_Comm_rank(comm, &rank));

    // load in the old alpha
    Vec keyValueVec;
    PetscCall(VecCreateMPI(comm, rank == 0 ? 1 : 0, 1, &keyValueVec));
    PetscCall(PetscObjectSetName((PetscObject)keyValueVec, name));
    PetscCall(VecLoad(keyValueVec, viewer));

    // Load in value
    if (rank == 0) {
        PetscInt index[1] = {0};
        PetscCall(VecGetValues(keyValueVec, 1, index, &value));
    }

    // Broadcast everywhere
    PetscCallMPI(MPI_Bcast(&value, 1, MPIU_SCALAR, 0, comm));
    PetscCall(VecDestroy(&keyValueVec));
    PetscFunctionReturn(0);
}

#include "mesh.h"

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, PetscBool simplex, PetscInt dimensions) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMPlexCreateBoxMesh(comm, dimensions, simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);

    ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);

    ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);

    // distribute the mesh
    {
        PetscPartitioner part;
        DM distributedMesh = NULL;

        /* Distribute mesh over processes */
        ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
        ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
        ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
        if (distributedMesh) {
            ierr = DMDestroy(dm);
            CHKERRQ(ierr);
            *dm = distributedMesh;
        }
    }

    PetscFunctionReturn(0);
}
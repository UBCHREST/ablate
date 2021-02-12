#include "mesh.h"

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, PetscBool simplex, PetscInt dimensions) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMPlexCreateBoxMesh(comm, dimensions, simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);

    ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);

    ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
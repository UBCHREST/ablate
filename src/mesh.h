#include <petsc.h>

#ifndef mesh_h
#define mesh_h

PETSC_EXTERN PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, PetscBool simplex, PetscInt dimensions);

#endif  // mesh_h
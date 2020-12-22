#include <petsc.h>

#ifndef mesh_h
#define mesh_h

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, PetscBool simplex, PetscInt dimensions);

#endif //mesh_h
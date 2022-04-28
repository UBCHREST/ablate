// These are functions that should probably make their way into PETSc at some point. Put them in here for now.
//
// Note that some of these functions (in particular DMGetFieldVec/DMRestoreFieldVec) may already be in ABLATE or have an equivalent

#include <string>
#include <vector>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>

namespace ablate::levelSet {
  class plex {
    private:
      PetscErrorCode DMPlexGetNeighborCells_Internal(DM dm, PetscInt p, PetscReal x0[3], PetscReal maxDist, PetscInt *nCells, PetscInt *cells[]);

    public:
      PetscErrorCode DMPlexGetNeighborCells(DM dm, PetscInt p, PetscInt levels, PetscReal h, PetscReal maxDist, PetscInt *nCells, PetscInt *cells[]);
      PetscErrorCode DMGetFieldVec(DM dm, Vec v, PetscInt field, PetscInt height, IS *is, Vec *subv);
      PetscErrorCode DMRestoreFieldVec(DM dm, Vec v, PetscInt field, PetscInt height, IS *is, Vec *subv);



  };

}

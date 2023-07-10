// These are functions that should probably make their way into PETSc at some point. Put them in here for now.
//
// Note that some of these functions (in particular DMGetFieldVec/DMRestoreFieldVec) may already be in ABLATE or have an equivalent

#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <string>
#include <vector>

PetscErrorCode DMPlexGetNeighbors(DM dm, PetscInt p, PetscInt levels, PetscReal maxDist, PetscInt minNumberCells, PetscBool useCells, PetscBool returnNeighborVeretices, PetscInt *nCells,
                                  PetscInt **cells);

PetscErrorCode DMPlexGetContainingCell(DM dm, PetscScalar *xyz, PetscInt *cell);

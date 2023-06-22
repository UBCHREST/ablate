// These are functions that should probably make their way into PETSc at some point. Put them in here for now.
//
// Note that some of these functions (in particular DMGetFieldVec/DMRestoreFieldVec) may already be in ABLATE or have an equivalent

#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <string>
#include <vector>

PetscErrorCode DMPlexGetNeighborCells(DM dm, PetscInt p, PetscInt levels, PetscReal maxDist, PetscInt minNumberCells, PetscBool useCells, PetscBool returnNeighborVeretices, PetscInt *nCells,
                                      PetscInt **cells);
PetscErrorCode DMGetFieldVec(DM dm, Vec v, PetscInt field, PetscInt height, IS *is, Vec *subv);
PetscErrorCode DMRestoreFieldVec(DM dm, Vec v, PetscInt field, PetscInt height, IS *is, Vec *subv);

PetscErrorCode DMPlexGetLineIntersection_2D_Internal(const PetscReal segmentA[], const PetscReal segmentB[], PetscReal intersection[], PetscBool *hasIntersection);
PetscErrorCode DMPlexGetLinePlaneIntersection_3D_Internal(const PetscReal segmentA[], const PetscReal segmentB[], const PetscReal segmentC[], PetscReal intersection[], PetscBool *hasIntersection);
PetscErrorCode DMPlaneVectors(DM dm, const PetscReal x0[], const PetscReal n[], const PetscReal offset, PetscReal segmentA[], PetscReal segmentB[]);

PetscErrorCode DMPlexGetContainingCell(DM dm, PetscScalar *xyz, PetscInt *cell);

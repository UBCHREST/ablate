#include "lsSupport.hpp"
#include <petsc/private/vecimpl.h>


using namespace ablate::levelSet;

PetscErrorCode plex::DMPlexGetNeighborCells_Internal(DM dm, PetscInt p, PetscReal x0[3], PetscReal maxDist, PetscInt *nCells, PetscInt *cells[]) {

  PetscInt        cStart, cEnd, vStart, vEnd;
  PetscInt        cl, nClosure, *closure = NULL;
  PetscInt        st, nStar, *star = NULL;
  PetscInt        n, list[100];  // As of right now just make it a list big enough to hold everything. There must be a better way of doing this.
  PetscInt        i, dim;
  PetscReal       x[3], dist;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);       // Range of cells
//  ierr = DMPlexGetHeightStratum(dm, 1, &vStart, &vEnd);CHKERRQ(ierr);       // Range of cells
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);        // Range of vertices

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);



  n = 0;
  ierr = DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &nClosure, &closure);CHKERRQ(ierr);
  for (cl = 0; cl < nClosure*2; cl += 2) {
    if (closure[cl] >= vStart && closure[cl] < vEnd){
      ierr = DMPlexGetTransitiveClosure(dm, closure[cl], PETSC_FALSE, &nStar, &star);CHKERRQ(ierr);
      for (st = 0; st< nStar*2; st += 2) {
        if( star[st] >= cStart && star[st] < cEnd){
          ierr = DMPlexComputeCellGeometryFVM(dm, star[st], NULL, x, NULL);CHKERRQ(ierr);
          dist = 0;
          for (i = 0; i < dim; ++i) {
            dist += PetscSqr(x0[i] - x[i]);
          }
          if (PetscSqrtReal(dist) <= maxDist) {
            list[n++] = star[st];
          }
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, closure[cl], PETSC_FALSE, &nStar, &star);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &nClosure, &closure);CHKERRQ(ierr);
  ierr = PetscSortRemoveDupsInt(&n, list);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, cells);CHKERRQ(ierr);
  ierr = PetscArraycpy(*cells, list, n);CHKERRQ(ierr);
  *nCells = n;



  PetscFunctionReturn(0);
}

PetscErrorCode plex::DMPlexGetNeighborCells(DM dm, PetscInt p, PetscInt levels, PetscReal h, PetscReal maxDist, PetscInt *nCells, PetscInt *cells[]) {
  PetscInt        numAdd, *addList;
  PetscInt        n = 0, n0, list[10000];
  PetscInt        l, i;
  PetscReal       x0[3];
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  if (levels > 0) {

    if (maxDist < -0.5) {
      maxDist = ((PetscReal)levels+1.0)*h;
    }

    ierr = DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL);CHKERRQ(ierr);

    ierr = DMPlexGetNeighborCells_Internal(dm, p, x0, maxDist, &n, &addList);CHKERRQ(ierr);
    ierr = PetscArraycpy(&list[0], addList, n);

    for (l = 1; l < levels; ++l) {
      n0 = n;
      for (i = 0; i < n0; ++i) {
        ierr = DMPlexGetNeighborCells_Internal(dm, list[i], x0, maxDist, &numAdd, &addList);CHKERRQ(ierr);
        ierr = PetscArraycpy(&list[n], addList, numAdd);
        n += numAdd;
        ierr = PetscFree(addList);
      }
      ierr = PetscSortRemoveDupsInt(&n, list);CHKERRQ(ierr);
    }

    ierr = PetscMalloc1(n, cells);CHKERRQ(ierr);
    ierr = PetscArraycpy(*cells, list, n);CHKERRQ(ierr);
    *nCells = n;
  }


  PetscFunctionReturn(0);
}



PetscErrorCode plex::DMGetFieldVec(DM dm, Vec v, PetscInt field, PetscInt height, IS *is, Vec *subv) {
  PetscSection    sectionLocal, sectionGlobal;
  PetscInt        cStart, cEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = DMPlexGetHeightStratum(dm, height, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &sectionLocal);CHKERRQ(ierr);

  ierr = PetscSectionGetField_Internal(sectionLocal, sectionGlobal, v, field, cStart, cEnd, is, subv);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode plex::DMRestoreFieldVec(DM dm, Vec v, PetscInt field, PetscInt height, IS *is, Vec *subv) {
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = VecRestoreSubVector(v, *is, subv);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#include "lsSupport.hpp"
#include <petsc/private/vecimpl.h>

// Return all cells which share an vertex or edge/face with a center cell
// dm - The mesh
// p - The cell to use
// maxDist - Maximum distance from p to consider adding
// useVertices - Should we include cells which share a vertex (TRUE) or an edge/face (FALSE)
// nCells - Number of cells found
// cells - The IDs of the cells found.
PetscErrorCode DMPlexGetNeighborCells_Internal(DM dm, PetscInt p, PetscReal maxDist, PetscBool useVertices, PetscInt *nCells, PetscInt *cells[]) {

  PetscInt        cStart, cEnd, vStart, vEnd;
  PetscInt        cl, nClosure, *closure = NULL;
  PetscInt        st, nStar, *star = NULL;
  PetscInt        n, list[100];  // As of right now just make it a list big enough to hold everything. There must be a better way of doing this.
  PetscInt        i, dim;
  PetscReal       x0[3], x[3], dist;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL);CHKERRQ(ierr); // Center of the cell-of-interest

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);       // Range of cells
  if (useVertices) {
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);      // Range of vertices
  }
  else {
    ierr = DMPlexGetHeightStratum(dm, 1, &vStart, &vEnd);CHKERRQ(ierr);     // Range of edges (2D) or faces (3D)
  }

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  n = 0;
  ierr = DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &nClosure, &closure);CHKERRQ(ierr); // All points associated with the cell
  for (cl = 0; cl < nClosure*2; cl += 2) {
    if (closure[cl] >= vStart && closure[cl] < vEnd){ // Only use the points corresponding to either a vertex or edge/face.
      ierr = DMPlexGetTransitiveClosure(dm, closure[cl], PETSC_FALSE, &nStar, &star);CHKERRQ(ierr); // Get all points using this vertex or edge/face.
      for (st = 0; st< nStar*2; st += 2) {
        if( star[st] >= cStart && star[st] < cEnd){   // If the point is a cell add it.
          ierr = DMPlexComputeCellGeometryFVM(dm, star[st], NULL, x, NULL);CHKERRQ(ierr); // Center of the candidate cell.
          dist = 0;
          for (i = 0; i < dim; ++i) {
            dist += PetscSqr(x0[i] - x[i]);
          }
          if (PetscSqrtReal(dist) <= maxDist) {   // Only add if the distance is within maxDist
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


// Return the list of neighboring cells to cell p using a combination of number of levels and maximum distance
// dm - The mesh
// levels - Number of neighboring cells to check
// maxDist - Maximum distance to include
// nCells - Number of neighboring cells
// cells - The list of neighboring cell IDs
PetscErrorCode DMPlexGetNeighborCells(DM dm, PetscInt p, PetscInt levels, PetscReal maxDist, PetscBool useVertices, PetscInt *nCells, PetscInt *cells[]) {
  PetscInt        numAdd, *addList;
  PetscInt        n = 0, n0, list[10000];
  PetscInt        l, i;
  PetscReal       h;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  if (levels > 0) {

    DMPlexGetMinRadius(dm, &h); // This returns the minimum distance from any cell centroid to a face.
    h *= 2.0;                   // Double it to get the grid spacing.

    // If the maximum distance isn't provided estimate it based on the number of levels
    if (maxDist < -0.5) {
      maxDist = ((PetscReal)levels+1.0)*h;
    }



    // Get one level of neighboring cells
    ierr = DMPlexGetNeighborCells_Internal(dm, p, maxDist, useVertices, &n, &addList);CHKERRQ(ierr);
    ierr = PetscArraycpy(&list[0], addList, n);

    for (l = 1; l < levels; ++l) {
      n0 = n;
      for (i = 0; i < n0; ++i) {
        ierr = DMPlexGetNeighborCells_Internal(dm, list[i], maxDist, useVertices, &numAdd, &addList);CHKERRQ(ierr);
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



PetscErrorCode DMGetFieldVec(DM dm, Vec v, PetscInt field, PetscInt height, IS *is, Vec *subv) {
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

PetscErrorCode DMRestoreFieldVec(DM dm, Vec v, PetscInt field, PetscInt height, IS *is, Vec *subv) {
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = VecRestoreSubVector(v, *is, subv);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


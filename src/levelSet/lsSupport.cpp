#include <petsc.h>

// Return the number of vertices associated with a given cell using the polytope.
PetscErrorCode DMPlexGetNumCellVertices(DM dm, const PetscInt p, PetscInt *nv) {
  DMPolytopeType ct;

  PetscFunctionBegin;

  PetscCall(DMPlexGetCellType(dm, p, &ct));

  switch (ct) {
    case DM_POLYTOPE_POINT:
      *nv = 1;
      break;
    case DM_POLYTOPE_SEGMENT:
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
      *nv = 2;
      break;
    case DM_POLYTOPE_TRIANGLE:
      *nv = 3;
      break;
    case DM_POLYTOPE_QUADRILATERAL:
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
    case DM_POLYTOPE_TETRAHEDRON:
      *nv = 4;
      break;
    case DM_POLYTOPE_PYRAMID:
      *nv = 5;
      break;
    case DM_POLYTOPE_TRI_PRISM:
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
      *nv = 6;
      break;
    case DM_POLYTOPE_HEXAHEDRON:
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
      *nv = 8;
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Cannot determine number of vertices for cell type %s", DMPolytopeTypes[ct]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);

}

PetscErrorCode DMPlexGetCellVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *vertOut[]) {
  PetscInt  vStart, vEnd;
  PetscInt  n;
  PetscInt  cl, nClosure, *closure = NULL;
  PetscInt  nv, *verts;

  PetscFunctionBegin;

  PetscCall(DMPlexGetNumCellVertices(dm, p, &nv));
  *nVerts = nv;

  PetscCall(DMGetWorkArray(dm, nv, MPIU_INT, vertOut));
  verts = *vertOut;

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));  // Range of vertices

  // This returns everything associated with the cell in the correct ordering
  PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &nClosure, &closure));

  n = 0;
  for (cl = 0; cl < nClosure * 2; cl += 2) {
    if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex
      verts[n++] = closure[cl];
    }
  }

  PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &nClosure, &closure));  // Restore the points

  PetscFunctionReturn(PETSC_SUCCESS);

}


PetscErrorCode DMPlexRestoreCellVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *vertOut[]) {

  PetscFunctionBegin;
  if (nVerts) *nVerts = 0;
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_INT, vertOut));
  PetscFunctionReturn(PETSC_SUCCESS);

}

// Return the coordinates of a list of vertices
PetscErrorCode DMPlexGetVertexCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]) {

  PetscInt    dim;
  DM          cdm;
  Vec         localCoordsVector;
  PetscScalar *coordsArray, *vcoords;
  PetscSection coordsSection;

  PetscFunctionBegin;

  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(DMGetCoordinateDM(dm, &cdm));

  PetscCall(DMGetWorkArray(dm, np*dim, MPIU_SCALAR, coords));

  PetscCall(DMGetCoordinatesLocal(dm, &localCoordsVector));
  PetscCall(VecGetArray(localCoordsVector, &coordsArray));
  PetscCall(DMGetCoordinateSection(dm, &coordsSection));

  for (PetscInt i = 0; i < np; ++i) {
    PetscCall(DMPlexPointLocalRef(cdm, pArray[i], coordsArray, &vcoords));
    for (PetscInt d = 0; d < dim; ++d ) {
      (*coords)[i*dim + d] = vcoords[d];
    }
  }

  PetscCall(VecRestoreArray(localCoordsVector, &coordsArray));

  PetscFunctionReturn(PETSC_SUCCESS);

}

PetscErrorCode DMPlexRestoreVertexCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscReal *coords[]) {
  PetscFunctionBegin;
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}


#include <petsc.h>

// Return the number of vertices associated with a given cell using the polytope.
PetscErrorCode DMPlexCellGetNumVertices(DM dm, const PetscInt p, PetscInt *nv) {
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

PetscErrorCode DMPlexCellGetVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *vertOut[]) {
  PetscInt  vStart, vEnd;
  PetscInt  n;
  PetscInt  cl, nClosure, *closure = NULL;
  PetscInt  nv, *verts;

  PetscFunctionBegin;

  PetscCall(DMPlexCellGetNumVertices(dm, p, &nv));
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


PetscErrorCode DMPlexCellRestoreVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *vertOut[]) {

  PetscFunctionBegin;
  if (nVerts) *nVerts = 0;
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_INT, vertOut));
  PetscFunctionReturn(PETSC_SUCCESS);

}




PetscErrorCode DMPlexVertexGetCells(DM dm, const PetscInt p, PetscInt *nCells, PetscInt *cellsOut[]) {
  PetscInt  cStart, cEnd;
  PetscInt  n;
  PetscInt  cl, nClosure, *closure = NULL;
  PetscInt  nc, *cells;

  PetscFunctionBegin;


  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // Range of cells

  // Everything using this vertex
  PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_FALSE, &nClosure, &closure));

  // Now get the number of cells
  nc = 0;
  for (cl = 0; cl < nClosure * 2; cl += 2) {
    if (closure[cl] >= cStart && closure[cl] < cEnd) {  // Only use the points corresponding to a vertex
      ++nc;
    }
  }
  *nCells = nc;

  // Get the work array to store the cell IDs
  PetscCall(DMGetWorkArray(dm, nc, MPIU_INT, cellsOut));
  cells = *cellsOut;

  // Now assign the cells
  n = 0;
  for (cl = 0; cl < nClosure * 2; cl += 2) {
    if (closure[cl] >= cStart && closure[cl] < cEnd) {  // Only use the points corresponding to a vertex
      cells[n++] = closure[cl];
    }
  }

  PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_FALSE, &nClosure, &closure));  // Restore the points

  PetscFunctionReturn(PETSC_SUCCESS);

}


PetscErrorCode DMPlexVertexRestoreCells(DM dm, const PetscInt p, PetscInt *nCells, PetscInt *cellsOut[]) {

  PetscFunctionBegin;
  if (nCells) *nCells = 0;
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_INT, cellsOut));
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

PetscErrorCode DMPlexRestoreVertexCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]) {
  PetscFunctionBegin;
  PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Wrapper for the field and non-field calls. Got annoyed having this all over the place.
PetscErrorCode xDMPlexPointLocalRef(DM dm, PetscInt p, PetscInt fID, PetscScalar *array, void *ptr) {
  PetscFunctionBegin;
  if (fID >= 0) PetscCall(DMPlexPointLocalFieldRef(dm, p, fID, array, ptr));
  else PetscCall(DMPlexPointLocalRef(dm, p, array, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode xDMPlexPointLocalRead(DM dm, PetscInt p, PetscInt fID, const PetscScalar *array, void *ptr) {
  PetscFunctionBegin;
  if (fID >= 0) PetscCall(DMPlexPointLocalFieldRead(dm, p, fID, array, ptr));
  else PetscCall(DMPlexPointLocalRead(dm, p, array, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexFaceCentroidOutwardNormal(DM dm, PetscInt cell, PetscInt face, PetscReal *centroid, PetscReal *n) {
  PetscFunctionBegin;

  // Get the cell center
  PetscReal x0[3];
  PetscCall(DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL));

  // Centroid and normal for face
  PetscReal faceCenter[3], normal[3];
  PetscCall(DMPlexComputeCellGeometryFVM(dm, face, NULL, faceCenter, normal));

  PetscInt dim;
  PetscCall(DMGetDimension(dm, &dim));

  if (centroid) {
    for( PetscInt d = 0; d < dim; ++d) centroid[d] = faceCenter[d];
  }

  if (n) {
    // Make sure the normal is aligned with the vector connecting the cell center and the face center
    PetscReal sgn = 0.0;
    for( PetscInt d = 0; d < dim; ++d) {
      sgn += normal[d]*(centroid[d] - x0[d]);
    }
    sgn = PetscSignReal(sgn);

    for( PetscInt d = 0; d < dim; ++d) {
      n[d] = sgn*normal[d];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}




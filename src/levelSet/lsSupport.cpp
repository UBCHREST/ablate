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

  // Centroid and normal for face/edge
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


// DMPlex Notes:
// cone: Get all elements one dimension lower that use the current point
// support: Get all elements one dimension higher that use the current point
// closure: Get all elements with a lower dimension that are associated with the current point
// star: Get all elements with a higher dimension that are associated with the current point

// Given a 2D mesh compute the surface area normal aligned with the input N.
// Note: In 2D there can never be more than two cells with a shared edge
static PetscErrorCode DMPlexEdgeSurfaceAreaNormal2D_Internal(DM dm, const PetscReal vCoords[], const PetscReal edgeCenter[], const PetscInt nCells, const PetscInt *cells, PetscReal N[]){
  PetscFunctionBegin;

  const PetscInt dim = 2;

  PetscReal edgeVertex[dim];
  for (PetscInt d = 0; d < dim; ++d) {
    // Vector from the edge center to the vertex. Used to determine the outward surface area normal
    edgeVertex[d] = vCoords[d] - edgeCenter[d];

    N[d] = 0.0;
  }

  for (PetscInt c = 0; c < nCells; ++c) {
    PetscReal cellCenter[dim];
    PetscReal edgeCell[dim];
    PetscCall(DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, cellCenter, NULL));

    // Vector from the edge center to the cell center
    for (PetscInt d = 0; d < dim; ++d) {
      edgeCell[d] = cellCenter[d] - edgeCenter[d];

    }

    // The sign of the cross-product between vectors edgeCell and edgeVertex is used to determine outward normal
    PetscReal sgn = PetscSignReal(edgeCell[0]*edgeVertex[1] - edgeCell[1]*edgeVertex[0]);

    N[0] += sgn*edgeCell[1];
    N[1] -= sgn*edgeCell[0];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode DMPlexEdgeSurfaceAreaNormal3D_Internal(DM dm, const PetscReal vCoords[], const PetscReal edgeCenter[], const PetscInt nFaces, const PetscInt *faces, PetscReal N[]){
  PetscFunctionBegin;

  const PetscInt dim = 3;
  PetscReal vertexEdge[dim];

  for (PetscInt d = 0; d < dim; ++d) {
    N[d] = 0.0;
    vertexEdge[d] = edgeCenter[d] - vCoords[d];
  }

  for (PetscInt f = 0; f < nFaces; ++f) {
    PetscInt nCells;
    const PetscInt *cells;
    PetscReal faceCenter[dim];

    PetscCall(DMPlexComputeCellGeometryFVM(dm, faces[f], NULL, faceCenter, NULL));

    PetscCall(DMPlexGetSupportSize(dm, faces[f], &nCells));
    PetscCall(DMPlexGetSupport(dm, faces[f], &cells));
    for (PetscInt c = 0; c < nCells; ++c) {
      PetscReal cellCenter[dim];
      PetscReal cellEdge[dim]; // Vector from cell center to edge center
      PetscReal cellFace[dim]; // Vector from cell center to face center
      PetscReal n[dim];        // Surface area normal. Needs to be corrected to align with the vector from the vertex to the edgeCenter

      PetscCall(DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, cellCenter, NULL));

      for (PetscInt d = 0; d < dim; ++d) {
        cellEdge[d] = edgeCenter[d] - cellCenter[d];
        cellFace[d] = faceCenter[d] - cellCenter[d];
      }


      // Surface area normal. Take into account the 0.5 needed to compute the area of the triangle
      n[0] = 0.5*(cellEdge[1]*cellFace[2] - cellEdge[2]*cellFace[1]);
      n[1] = 0.5*(cellEdge[2]*cellFace[0] - cellEdge[0]*cellFace[2]);
      n[2] = 0.5*(cellEdge[0]*cellFace[1] - cellEdge[1]*cellFace[0]);

      // The sign of the dot product of surface area normal and vector connecting vertex to edge center gives the correction
      PetscReal sgn = PetscSignReal(vertexEdge[0]*n[0] + vertexEdge[1]*n[1] + vertexEdge[2]*n[2]);

      for (PetscInt d = 0; d < dim; ++d) {
        N[d] += sgn*n[d];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compute the edge surface area normal as defined in Morgan and Waltz with respect to a given vertex and an edge center
// NOTE: This does NOT check if the vertex and edge are actually associated with each other.
PetscErrorCode DMPlexEdgeSurfaceAreaNormal(DM dm, const PetscInt v, const PetscInt e, PetscReal N[]){
  PetscFunctionBegin;

  PetscReal      edgeCenter[3], vCoords[3];
  PetscInt       dim;
  PetscInt       nFace;
  const PetscInt *faces;

  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(DMPlexComputeCellGeometryFVM(dm, v, NULL, vCoords, NULL));
  PetscCall(DMPlexComputeCellGeometryFVM(dm, e, NULL, edgeCenter, NULL));

  // Get all of the cells(2D) or faces(3D) associated with this edge
  PetscCall(DMPlexGetSupportSize(dm, e, &nFace));
  PetscCall(DMPlexGetSupport(dm, e, &faces));

  switch (dim) {
    case 1:
      N[0] = edgeCenter[0] - vCoords[0]; // In 1D this is simply the vector connecting the vertex to the edge center
      break;
    case 2:
      DMPlexEdgeSurfaceAreaNormal2D_Internal(dm, vCoords, edgeCenter, nFace, faces, N);
      break;
    case 3:
      DMPlexEdgeSurfaceAreaNormal3D_Internal(dm, vCoords, edgeCenter, nFace, faces, N);
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "DMPlexEdgeSurfaceAreaNormal can not handle dimensions of %d", dim);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Calculate the volume of the CV surrounding a vertex
PetscErrorCode DMPlexVertexControlVolume(DM dm, const PetscInt v, PetscReal *vol) {

  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt nStar, *star = NULL;
  PetscInt dim;

  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // Range of cells


  PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star)); // Everything using this vertex

  *vol = 0.0;

  for (PetscInt st = 0; st < nStar * 2; st += 2) {
    if (star[st]>=cStart && star[st]<cEnd) {
      PetscReal cellVol;
      PetscInt nCorners;

      PetscCall(DMPlexComputeCellGeometryFVM(dm, star[st], &cellVol, NULL, NULL));

      // The number of corners a cell can be divided into equals the number of vertices associated with that cell
      PetscCall(DMPlexCellGetNumVertices(dm, star[st], &nCorners));

      *vol += cellVol/nCorners;
    }
  }

  PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star)); // Everything using this edge

  PetscFunctionReturn(PETSC_SUCCESS);

}


// Compute the finite-difference derivative approximation using the Eq. (11) from "3D level set methods for evolving fronts on tetrahedral
//    meshes with adaptive mesh refinement", by Morgan and Waltz, JCP 336 (2017) 492-512.
//   This should be second-order accurate for both triangles and quads
PetscErrorCode DMPlexVertexDerivative(DM dm, const PetscInt v, Vec data, PetscInt fID, PetscScalar g[]) {
  PetscFunctionBegin;


  PetscInt nEdge;
  const PetscInt *edge;
  const PetscScalar *dataArray;
  PetscInt vStart, vEnd;
  PetscInt dim;

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));  // Range of vertices
  PetscCheck(v >= vStart && v < vEnd, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "DMPlexVertexDerivative must have a valid vertex as input.");

  PetscCall(DMGetDimension(dm, &dim));

  PetscCheck(dim>0 && dim<4, PETSC_COMM_SELF, PETSC_ERR_SUP, "DMPlexVertexDerivative does not support a DM of dimension %d", dim);

  for (PetscInt d = 0; d < dim; ++d) {
    g[d] = 0.0;
  }

  PetscCall(VecGetArrayRead(data, &dataArray));

  // Get the edges associated with the vertex
  PetscCall(DMPlexGetSupportSize(dm, v, &nEdge));
  PetscCall(DMPlexGetSupport(dm, v, &edge));
  for (PetscInt e = 0; e < nEdge; ++e) {

    // Surface area normal
    PetscScalar N[3];
    PetscCall(DMPlexEdgeSurfaceAreaNormal(dm, v, edge[e], N));

    // Get vertices associated with this edge
    const PetscInt *verts;
    PetscCall(DMPlexGetCone(dm, edge[e], &verts));

    PetscReal *val, edgeVal;
    PetscCall(xDMPlexPointLocalRead(dm, verts[0], fID, dataArray, &val));

    edgeVal = 0.5*(*val);

    PetscCall(xDMPlexPointLocalRead(dm, verts[1], fID, dataArray, &val));
    edgeVal += 0.5*(*val);

    for (PetscInt d = 0; d < dim; ++d) {
      g[d] += edgeVal*N[d];
    }
  }

  PetscReal cvVol;
  PetscCall(DMPlexVertexControlVolume(dm, v, &cvVol));
  for (PetscInt d = 0; d < dim; ++d) {
    g[d] /= cvVol;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


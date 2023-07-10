PetscErrorCode DMPlexCellGetNumVertices(DM dm, const PetscInt p, PetscInt *nv);

PetscErrorCode DMPlexCellGetVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *verts[]);
PetscErrorCode DMPlexCellRestoreVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *vertOut[]);

PetscErrorCode DMPlexGetVertexCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]);
PetscErrorCode DMPlexRestoreVertexCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]);

PetscErrorCode DMPlexVertexRestoreCells(DM dm, const PetscInt p, PetscInt *nCells, PetscInt *cellsOut[]);
PetscErrorCode DMPlexVertexGetCells(DM dm, const PetscInt p, PetscInt *nCells, PetscInt *cellsOut[]);

PetscErrorCode xDMPlexPointLocalRef(DM dm, PetscInt p, PetscInt fID, PetscScalar *array, void *ptr);
PetscErrorCode xDMPlexPointLocalRead(DM dm, PetscInt p, PetscInt fID, const PetscScalar *array, void *ptr);

PetscErrorCode DMPlexFaceCentroidOutwardAreaNormal(DM dm, PetscInt cell, PetscInt face, PetscReal *centroid, PetscReal *n);

/**
  * Compute the gradient of a field defined over vertices at a vertex
  * @param dm - The DM of the data stored in vec
  * @param v - Vertex where to compute the gradient
  * @param data - Vector containing the data
  * @param fID - Field ID of the data to take the gradient of
  * @param g - The gradient at c
  */
PetscErrorCode DMPlexVertexGradFromVertex(DM dm, const PetscInt v, Vec data, PetscInt fID, PetscScalar g[]);


/**
  * Compute the gradient of a field defined over cells at a vertex
  * @param dm - The DM of the data stored in vec
  * @param v - Vertex where to compute the gradient
  * @param data - Vector containing the data
  * @param fID - Field ID of the data to take the gradient of
  * @param g - The gradient at c
  */
PetscErrorCode DMPlexVertexGradFromCell(DM dm, const PetscInt v, Vec data, PetscInt fID, PetscScalar g[]);

/**
  * Compute the gradient of a field defined over vertices at a cell center
  * @param dm - The DM of the data stored in vec
  * @param c - Cell where to compute the gradient
  * @param data - Vector containing the data
  * @param fID - Field ID of the data to take the gradient of
  * @param g - The gradient at c
  */
PetscErrorCode DMPlexCellGradFromVertex(DM dm, const PetscInt c, Vec data, PetscInt fID, PetscScalar g[]);

/**
  * Returns all DMPlex points at a given depth which are common between two DMPlex points. For example, if p1 is a cell and p2 is a vertex on the cell with depth=1 this will
  *   return the edges common to both p1 and p2
  * @param dm - The mesh
  * @param p1 - ID of the first point
  * @param p2 - ID of the second point
  * @param depth - The depth of the common point(s) to return
  * @param nPoints - Number of common points
  * @param points - The common points
  */
PetscErrorCode DMPlexGetCommonPoints(DM dm, const PetscInt p1, const PetscInt p2, const PetscInt depth, PetscInt *nPoints, PetscInt *points[]);
PetscErrorCode DMPlexRestoreCommonPoints(DM dm, const PetscInt p1, const PetscInt p2, const PetscInt depth, PetscInt *nPoints, PetscInt *points[]);

/**
  * The outward surface area normal of the corner of a cell with respect to a given vertex of said cell
  * @param dm - The mesh
  * @param v - ID of the vertes
  * @param c - ID of the cell with the vertex v
  * @param N - Outward surface area normal
  */
PetscErrorCode DMPlexCornerSurfaceAreaNormal(DM dm, const PetscInt v, const PetscInt c, PetscReal N[]);



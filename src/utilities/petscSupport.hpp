// These are functions that should probably make their way into PETSc at some point. Put them in here for now.

#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <string>
#include <vector>

/**
 * Return the list of neighboring cells/vertices to cell p using a combination of number of levels and maximum distance
 * dm - The mesh
 * maxLevels - Number of neighboring cells/vertices to check
 * maxDist - Maximum distance to include
 * numberCells - The number of cells/vertices to return.
 * useCells -
 * nCells - Number of neighboring cells/vertices
 * cells - The list of neighboring cell/vertices IDs
 *
 * Note: The intended use is to use either maxLevels OR maxDist OR minNumberCells.
 */
PetscErrorCode DMPlexRestoreNeighbors(DM dm, PetscInt p, PetscInt maxLevels, PetscReal maxDist, PetscInt numberCells, PetscBool useCells, PetscBool returnNeighborVertices, PetscInt *nCells,
                                  PetscInt **cells);
PetscErrorCode DMPlexGetNeighbors(DM dm, PetscInt p, PetscInt levels, PetscReal maxDist, PetscInt minNumberCells, PetscBool useCells, PetscBool returnNeighborVertices, PetscInt *nCells,
                                  PetscInt **cells);

/**
 * Return the cell containing a point
 * @param dm - The mesh
 * @param xyz - Location
 * @param cell - Cell containing the location. It will return -1 if xyz is not in the local portion of the DM.
 */
PetscErrorCode DMPlexGetContainingCell(DM dm, PetscScalar *xyz, PetscInt *cell);

/**
 * Get the number of vertices for a given cell
 * @param dm - The mesh
 * @param p - Vertex ID
 * @param nv - Number of vertices
 */
PetscErrorCode DMPlexCellGetNumVertices(DM dm, const PetscInt p, PetscInt *nv);

/**
 * Get/Restore all vertices associated with a cell
 * @param dm - The mesh
 * @param p - Cell ID
 * @param nCells - Number of vertices
 * @param cells - List of the vertices
 */
PetscErrorCode DMPlexCellGetVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *verts[]);
PetscErrorCode DMPlexCellRestoreVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *vertOut[]);

/**
 * Get/Restore the coordinates of a list of vertices
 * @param dm - The mesh
 * @param np - Number of vertices
 * @param pArray - Array of verteices
 * @param coords - Array of coordinates, given as [x0 y0 z0 x1 y1 z1 .....]
 */
PetscErrorCode DMPlexVertexGetCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]);
PetscErrorCode DMPlexVertexRestoreCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]);

/**
 * Get/Restore all cells associated with a vertes
 * @param dm - The mesh
 * @param p - Vertex ID
 * @param nCells - Number of cells which use this vertex
 * @param cells - List of the cells
 */
PetscErrorCode DMPlexVertexGetCells(DM dm, const PetscInt p, PetscInt *nCells, PetscInt *cells[]);
PetscErrorCode DMPlexVertexRestoreCells(DM dm, const PetscInt p, PetscInt *nCells, PetscInt *cells[]);

// Helper functions due to getting annoyed with having the if-statement for fID
PetscErrorCode xDMPlexPointLocalRef(DM dm, PetscInt p, PetscInt fID, PetscScalar *array, void *ptr);
PetscErrorCode xDMPlexPointLocalRead(DM dm, PetscInt p, PetscInt fID, const PetscScalar *array, void *ptr);

/**
 * Compute the gradient of a field defined over vertices at a vertex
 * @param dm - The DM of the data stored in vec
 * @param v - Vertex where to compute the gradient
 * @param data - Vector containing the data
 * @param fID - Field ID of the data to take the gradient of
 * @param offset - If fID points to a vector then indicate which component to use
 * @param g - The gradient at c
 */
PetscErrorCode DMPlexVertexGradFromVertex(DM dm, const PetscInt v, Vec data, PetscInt fID, PetscInt offset, PetscScalar g[]);

/**
 * Compute the gradient of a field defined over cells at a vertex
 * @param dm - The DM of the data stored in vec
 * @param v - Vertex where to compute the gradient
 * @param data - Vector containing the data
 * @param fID - Field ID of the data to take the gradient of
 * @param offset - If fID points to a vector then indicate which component to use
 * @param g - The gradient at c
 */
PetscErrorCode DMPlexVertexGradFromCell(DM dm, const PetscInt v, Vec data, PetscInt fID, PetscInt offset, PetscScalar g[]);

/**
 * Compute the gradient of a field defined over vertices at a cell center
 * @param dm - The DM of the data stored in vec
 * @param c - Cell where to compute the gradient
 * @param data - Vector containing the data
 * @param fID - Field ID of the data to take the gradient of
 * @param offset - If fID points to a vector then indicate which component to use
 * @param g - The gradient at c
 */
PetscErrorCode DMPlexCellGradFromVertex(DM dm, const PetscInt c, Vec data, PetscInt fID, PetscInt offset, PetscScalar g[]);


/**
 * Compute the corner surface area normal as defined in Morgan and Waltz with respect to a given vertex and a cell
 *    NOTE: This does NOT check if the vertex and cell are actually associated with each other.
 * @param dm - The mesh
 * @param v - The vertex associated with cell c
 * @param c - The cell associated with vertex v
 * @param N - The corner surface area normal
 */
PetscErrorCode DMPlexCornerSurfaceAreaNormal(DM dm, const PetscInt v, const PetscInt c, PetscReal N[]);



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
 * Return all values in sorted array a that are NOT in sorted array b. This is done in-place on array a.
 * Inputs:
 *    na - Size of sorted array b[]
 *    a - Array of integers
 *    nb - Size of sorted array a[]
 *    b - Array or integers
 *
 * Outputs:
 *    nb - Number of integers in b but not in a
 *    b - All integers in b but not in a
 */
PetscErrorCode PetscSortedArrayComplement(const PetscInt na, const PetscInt a[], PetscInt *nb, PetscInt b[]);

/**
 * Return all common values in sorted arrays a and b. This is done in-place on array b.
 * Inputs:
 *    na - Size of sorted array a[]
 *    a - Array or integers
 *    nb - Size of sorted array b[]
 *    b - Array of integers
 *
 * Outputs:
 *    nb - Number of integers in a and b
 *    b - All integers in a and b
 */
PetscErrorCode PetscSortedArrayCommon(const PetscInt na, const PetscInt a[], PetscInt *nb, PetscInt b[]);

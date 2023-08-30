#include "petscSupport.hpp"
#include <petsc/private/vecimpl.h>

/**
 * Return the cell containing the location xyz
 * Inputs:
 *  dm - The mesh
 *  xyz - Array containing the point
 *
 * Outputs:
 *  cell - The cell containing xyz. Will return -1 if this point is not in the local part of the DM
 *
 * Note: This is adapted from DMInterpolationSetUp. If the cell containing the point is a ghost cell then this will return -1.
 *        If the point is in the upper corner of the domain it will not be able to find the containing cell.
 *        This is also SLOW and needs -dm_plex_hash_location activated.
 */
PetscErrorCode DMPlexGetContainingCell(DM dm, const PetscScalar *xyz, PetscInt *cell) {
    PetscSF cellSF = NULL;
    Vec pointVec;
    PetscInt dim;
    const PetscSFNode *foundCells;
    const PetscInt *foundPoints;
    PetscInt numFound;

    PetscFunctionBegin;

    PetscCall(DMGetDimension(dm, &dim));

    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, xyz, &pointVec));

    PetscCall(DMLocatePoints(dm, pointVec, DM_POINTLOCATION_NONE, &cellSF));

    PetscCall(PetscSFGetGraph(cellSF, NULL, &numFound, &foundPoints, &foundCells));

    if (numFound == 0) {
        *cell = -1;
    } else {
        *cell = foundCells[0].index;
    }

    PetscCall(PetscSFDestroy(&cellSF));

    PetscCall(VecDestroy(&pointVec));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * Return the cell with a centroid of xyz
 * Inputs:
 *  dm - The mesh
 *  xyz - The centroid
 *  eps - The tolerance or `PETSC_DEFAULT`
 *
 * Outputs:
 *  cell - The cell with a centroid of xyz. Will return -1 if this point is not in the local part of the DM
 *
 * The tolerance is interpreted as the maximum Euclidean (L2) distance of the sought point from the specified coordinates.
 *
 * Complexity of this function is currently O(n) with n the number of vertices in the local mesh.

.seealso: `DMPLEX`, `DMPlexCreate()`, `DMGetCoordinatesLocal()`, `DMPlexFindVertices`
@*/
PetscErrorCode DMPlexFindCell(DM dm, const PetscScalar *xyz, PetscReal eps, PetscInt *cell) {
    //  PetscInt           c, cdim, i, j, o, p, pStart, pEnd;
    //  PetscInt           npoints;
    //  const PetscScalar *coord;
    //  Vec                allCoordsVec;
    //  const PetscScalar *allCoords;
    //  PetscInt          *dagPoints;

    DM cellGeomDm;
    PetscInt cStart, cEnd, dim;
    const PetscScalar *cellGeomArray;
    Vec cellGeomVec;

    PetscFunctionBegin;

    PetscCall(DMGetDimension(dm, &dim));

    if (eps < 0) eps = PETSC_SQRT_MACHINE_EPSILON;

    PetscCall(DMPlexGetDataFVM(dm, nullptr, &cellGeomVec, nullptr, nullptr));
    PetscCall(VecGetArrayRead(cellGeomVec, &cellGeomArray));
    PetscCall(VecGetDM(cellGeomVec, &cellGeomDm));

    *cell = -1;

    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscFVCellGeom *cellGeom;
        PetscCall(DMPlexPointLocalRead(cellGeomDm, c, cellGeomArray, &cellGeom));

        PetscReal norm = 0.0;
        for (PetscInt d = 0; d < dim; d++) norm += PetscRealPart(PetscSqr(xyz[d] - cellGeom->centroid[d]));
        norm = PetscSqrtReal(norm);
        if (norm <= eps) {
            *cell = c;
            break;
        }
    }

    PetscCall(VecRestoreArrayRead(cellGeomVec, &cellGeomArray));

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * Return all cells which share an vertex or edge/face with a center cell
 * Inputs:
 *    dm - The mesh
 *    x0 - The location of the true center cell
 *    p - The cell to get the neighboors of
 *    maxDist - Maximum distance from p to consider adding
 *    useFace - Should we include cells which share a vertex (FALSE) or an edge/face (TRUE)
 *
 * Outputs:
 *    nCells - Number of cells found
 *    cells - The IDs of the cells found.
 */
static PetscErrorCode DMPlexGetNeighborCells_Internal(DM dm, PetscReal x0[3], PetscInt p, PetscReal maxDist, PetscBool useFace, PetscInt *nCells, PetscInt *cells[]) {
    PetscInt cStart, cEnd, vStart, vEnd;
    PetscInt cl, nClosure, *closure = NULL;
    PetscInt st, nStar, *star = NULL;
    PetscInt n, list[10000];  // To avoid memory reallocation just make the list bigger than it would ever need to be. Will look at doing something else in the future.
    PetscInt i, dim;
    PetscReal x[3], dist;

    PetscFunctionBegin;

    PetscCall(DMGetDimension(dm, &dim));  // The dimension of the grid

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // Range of cells

    if (useFace) {
        PetscCall(DMPlexGetHeightStratum(dm, 1, &vStart, &vEnd));  // Range of edges (2D) or faces (3D)
    } else {
        PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));  // Range of vertices
    }

    n = 0;
    PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &nClosure, &closure));  // All points associated with the cell

    maxDist = PetscSqr(maxDist) + PETSC_MACHINE_EPSILON;  // So we don't need PetscSqrtReal in the check

    for (cl = 0; cl < nClosure * 2; cl += 2) {
        if (closure[cl] >= vStart && closure[cl] < vEnd) {                                       // Only use the points corresponding to either a vertex or edge/face.
            PetscCall(DMPlexGetTransitiveClosure(dm, closure[cl], PETSC_FALSE, &nStar, &star));  // Get all points using this vertex or edge/face.

            for (st = 0; st < nStar * 2; st += 2) {
                if (star[st] >= cStart && star[st] < cEnd) {                               // If the point is a cell add it.
                    PetscCall(DMPlexComputeCellGeometryFVM(dm, star[st], NULL, x, NULL));  // Center of the candidate cell.
                    dist = 0.0;
                    for (i = 0; i < dim; ++i) {  // Compute the distance so that we can check if it's within the required distance.
                        dist += PetscSqr(x0[i] - x[i]);
                    }

                    if (dist <= maxDist && star[st] != p) {  // Only add if the distance is within maxDist and it's not the center cell
                        list[n++] = star[st];
                    }
                }
            }

            PetscCall(DMPlexRestoreTransitiveClosure(dm, closure[cl], PETSC_FALSE, &nStar, &star));  // Restore the points
        }
    }

    PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &nClosure, &closure));  // Restore the points
    PetscCall(PetscSortRemoveDupsInt(&n, list));                                        // Cleanup the list
    if (!(*cells)) PetscCall(PetscMalloc1(n, cells));                                   // Allocate the output
    PetscCall(PetscArraycpy(*cells, list, n));                                          // Copy the cell list
    *nCells = n;                                                                        // Set the number of cells for output

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * Return all vertices which share a cell or an edge/face with a desired vertex/cell
 * Inputs:
 *    dm - The mesh
 *    x0 - The location of the true desired vertex
 *    p - The vertex to get the neighboors of
 *    maxDist - Maximum distance from p to consider adding
 *    useCells - Should we include vertices which share a cell (TRUE) or an edge/face (FALSE)
 *
 * Outputs:
 *    nVertices - Number of vertices found
 *    vertices- The IDs of the vertices found.
 */
static PetscErrorCode DMPlexGetNeighborVertices_Internal(DM dm, PetscReal x0[3], PetscInt p, PetscReal maxDist, PetscBool useCells, PetscInt *nVertices, PetscInt *vertices[]) {
    PetscInt cStart, cEnd, vStart, vEnd;
    PetscInt cl, nClosure, *closure = NULL;
    PetscInt st, nStar, *star = NULL;
    PetscInt n, list[10000];  // To avoid memory reallocation just make the list bigger than it would ever need to be. Will look at doing something else in the future.
    PetscInt i, dim, i_x;
    PetscReal dist;
    Vec coords;
    PetscReal *coordsArray;

    PetscFunctionBegin;

    PetscCall(DMGetDimension(dm, &dim));  // The dimension of the grid

    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));  // Range of vertices

    if (useCells) {
        PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // Range of cells
    } else {
        PetscCall(DMPlexGetHeightStratum(dm, 1, &cStart, &cEnd));  // Range of edges (2D) or faces (3D)
    }

    n = 0;
    PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_FALSE, &nStar, &star));  // All points associated with the vertex

    maxDist = PetscSqr(maxDist) + PETSC_MACHINE_EPSILON;  // So we don't need PetscSqrtReal in the check
    PetscCall(DMGetCoordinatesLocal(dm, &coords));        // Get all the vertices coordinates
    PetscCall(VecGetArray(coords, &coordsArray));         // Copy the quantities in coords vector and paste them to the coordsArray vector

    for (st = 0; st < nStar * 2; st += 2) {
        if (star[st] >= cStart && star[st] < cEnd) {                                               // Only use the points corresponding to either a vertex or edge/face.
            PetscCall(DMPlexGetTransitiveClosure(dm, star[st], PETSC_TRUE, &nClosure, &closure));  // Get all points using this vertex or edge/face.

            for (cl = 0; cl < nClosure * 2; cl += 2) {
                if (closure[cl] >= vStart && closure[cl] < vEnd) {  // If the point is a vertex add it.
                    dist = 0.0;
                    i_x = (closure[cl] - vStart) * dim;
                    for (i = 0; i < dim; ++i) {  // Compute the distance so that we can check if it's within the required distance.
                        dist += PetscSqr(x0[i] - coordsArray[i_x + i]);
                    }

                    if (dist <= maxDist && closure[cl] != p) {  // Only add if the distance is within maxDist and it's not the main vertex
                        list[n++] = closure[cl];
                    }
                }
            }

            PetscCall(DMPlexRestoreTransitiveClosure(dm, star[cl], PETSC_TRUE, &nClosure, &closure));  // Restore the points
        }
    }

    PetscCall(VecRestoreArray(coords, &coordsArray));                             // Restore the coordsArray
    PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &nStar, &star));  // Restore the points
    PetscCall(PetscSortRemoveDupsInt(&n, list));                                  // Cleanup the list
    if (!(*vertices)) PetscCall(PetscMalloc1(n, vertices));                       // Allocate the output
    PetscCall(PetscArraycpy(*vertices, list, n));                                 // Copy the vertex list
    *nVertices = n;                                                               // Set the number of vertices for output

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexRestoreNeighbors(DM dm, PetscInt p, PetscInt maxLevels, PetscReal maxDist, PetscInt numberCells, PetscBool useCells, PetscBool returnNeighborVertices, PetscInt *nCells,
                                      PetscInt **cells) {
    PetscFunctionBegin;
    if (nCells) *nCells = 0;
    PetscCall(DMRestoreWorkArray(dm, 0, MPIU_INT, cells));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexGetNeighbors(DM dm, PetscInt p, PetscInt maxLevels, PetscReal maxDist, PetscInt numberCells, PetscBool useCells, PetscBool returnNeighborVertices, PetscInt *nCells,
                                  PetscInt **cells) {
    const PetscInt maxLevelListSize = 10000;
    const PetscInt maxListSize = 100000;
    PetscInt numNew, nLevelList[2];
    PetscInt *addList = NULL, levelList[2][maxLevelListSize], currentLevelLoc, prevLevelLoc;
    PetscInt n = 0, list[maxListSize];
    PetscInt l, i, cte;
    PetscScalar x0[3];
    PetscInt type = 0;  // 0: numberCells, 1: maxLevels, 2: maxDist

    PetscFunctionBegin;

    PetscCheck(
        ((maxLevels > 0) + (maxDist > 0) + (numberCells > 0)) == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Only one of maxLevels, maxDist, and minNumberCells can be set. The others whould be <0.");

    // Use minNumberCells if provided
    if (numberCells > 0) {
        maxLevels = PETSC_MAX_INT;
        maxDist = PETSC_MAX_REAL;
        type = 0;
    } else if (maxLevels > 0) {
        numberCells = PETSC_MAX_INT;
        maxDist = PETSC_MAX_REAL;
        type = 1;
    } else {  // Must be maxDist
        maxLevels = PETSC_MAX_INT;
        numberCells = PETSC_MAX_INT;
        type = 2;
    }

    PetscCall(DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL));  // Center of the cell-of-interest

    // Declare the internal function pointer
    PetscErrorCode (*neighborFunc)(DM, PetscReal[3], PetscInt, PetscReal, PetscBool, PetscInt *, PetscInt **);

    // Determine which internal function to call in while loop; if retutnNeighborVertices is false, the function returns the neighboring cells, and for true value, it returns vertices.
    l = 0;  // Current level
    if (returnNeighborVertices == PETSC_FALSE) {
        cte = 0;
        neighborFunc = &DMPlexGetNeighborCells_Internal;
        // Start with only the center cell
        list[0] = p;
        n = nLevelList[0] = 1;
        levelList[0][0] = p;
        currentLevelLoc = 0;
    } else {
        cte = 1;
        neighborFunc = &DMPlexGetNeighborVertices_Internal;
        // get first level vertices for p and start the while loop from those vertices
        PetscInt *closure = NULL;
        PetscInt closureSize;
        DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &closureSize, &closure);
        PetscInt start, end;
        DMPlexGetDepthStratum(dm, 0, &start, &end);  // Get the range of vertex indices
        for (PetscInt ii = 0; ii < closureSize * 2; ii += 2) {
            PetscInt point = closure[ii];
            if (point >= start && point < end) {
                // point is a vertex of the cell
                list[n] = point;
                levelList[0][n] = point;
                n++;
            }
        }
        nLevelList[0] = n;
        currentLevelLoc = 0;
        DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &closureSize, &closure);
        PetscCall(PetscIntSortSemiOrdered(nLevelList[0], levelList[0]));
        PetscCall(PetscIntSortSemiOrdered(nLevelList[0], list));
    }

    // When the number of cells added at a particular level is zero then terminate the loop. This is for the case where
    // maxLevels is set very large but all cells within the maximum distance have already been found.
    PetscCall(PetscMalloc1(maxLevelListSize, &addList));
    while (l < maxLevels && n < numberCells && nLevelList[currentLevelLoc] > 0) {
        ++l;

        // This will alternate between the levelLists
        currentLevelLoc = l % 2;
        prevLevelLoc = (l + 1) % 2;

        nLevelList[currentLevelLoc] = 0;
        for (i = 0; i < nLevelList[prevLevelLoc]; ++i) {  // Iterate over each of the locations on the prior level
            PetscCall((*neighborFunc)(dm, x0, levelList[prevLevelLoc][i], maxDist, useCells, &numNew, &addList));

            PetscCheck((nLevelList[currentLevelLoc] + numNew) < maxLevelListSize,
                       PETSC_COMM_SELF,
                       PETSC_ERR_ARG_INCOMP,
                       "Requested level list size has exceeded the maximum possible in DMPlexGetNeighborCells.");

            PetscCall(PetscArraycpy(&levelList[currentLevelLoc][nLevelList[currentLevelLoc]], addList, numNew));
            nLevelList[currentLevelLoc] += numNew;
        }
        PetscCall(PetscSortRemoveDupsInt(&nLevelList[currentLevelLoc], levelList[currentLevelLoc]));

        // This removes any cells which are already in the list. Not point in re-doing the search for those.
        PetscCall(PetscSortedArrayComplement(n, list, &nLevelList[currentLevelLoc], levelList[currentLevelLoc]));

        PetscCheck((n + nLevelList[currentLevelLoc]) < maxListSize, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Requested list size has exceeded the maximum possible in DMPlexGetNeighborCells.");

        PetscCall(PetscArraycpy(&list[n], levelList[currentLevelLoc], nLevelList[currentLevelLoc]));
        n += nLevelList[currentLevelLoc];

        PetscCall(PetscIntSortSemiOrdered(n, list));
    }

    PetscCall(PetscFree(addList));

    if (type == 0 && cte == 0) {
        // Now only include the the numberCells closest cells
        PetscScalar x[3];
        PetscReal *dist;
        PetscInt j, dim;
        PetscCall(DMGetDimension(dm, &dim));  // The dimension of the grid
        PetscCall(PetscMalloc1(n, &dist));
        for (i = 0; i < n; ++i) {
            PetscCall(DMPlexComputeCellGeometryFVM(dm, list[i], NULL, x, NULL));  // Center of the cell-of-interest
            dist[i] = 0.0;
            for (j = 0; j < dim; ++j) {  // Compute the distance so that we can check if it's within the required distance.
                dist[i] += PetscSqr(x0[j] - x[j]);
            }
        }
        PetscCall(PetscSortRealWithArrayInt(n, dist, list));
        PetscCall(PetscFree(dist));
    } else if (type == 0 && cte == 1) {
        // Now only include the the numberCells closest cells
        PetscReal *dist;
        PetscInt j, dim, i_x, vStart;
        Vec coords;
        PetscReal *coordsArray;
        PetscCall(DMGetDimension(dm, &dim));  // The dimension of the grid
        PetscCall(PetscMalloc1(n, &dist));
        PetscCall(DMGetCoordinatesLocal(dm, &coords));           // Get all the vertices coordinates
        PetscCall(VecGetArray(coords, &coordsArray));            // Copy the quantities in coords vector and paste them to the coordsArray vector
        PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, NULL));  // Range of vertices
        for (i = 0; i < n; ++i) {
            dist[i] = 0.0;
            i_x = (list[i] - vStart) * dim;
            for (j = 0; j < dim; ++j) {  // Compute the distance so that we can check if it's within the required distance.
                dist[i] += PetscSqr(x0[j] - coordsArray[i_x + j]);
            }
        }
        PetscCall(PetscSortRealWithArrayInt(n, dist, list));
        PetscCall(PetscFree(dist));
    } else {
        numberCells = n;
    }
    PetscCall(DMGetWorkArray(dm, numberCells, MPIU_INT, cells));
    PetscCall(PetscArraycpy(*cells, list, numberCells));
    PetscCall(PetscIntSortSemiOrdered(numberCells, *cells));
    *nCells = numberCells;

    PetscFunctionReturn(PETSC_SUCCESS);
}

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
    PetscInt vStart, vEnd;
    PetscInt n;
    PetscInt cl, nClosure, *closure = NULL;
    PetscInt nv, *verts;

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
    PetscInt cStart, cEnd;
    PetscInt n;
    PetscInt cl, nClosure, *closure = NULL;
    PetscInt nc, *cells;

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
PetscErrorCode DMPlexVertexGetCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]) {
    PetscInt dim;
    DM cdm;
    Vec localCoordsVector;
    PetscScalar *coordsArray, *vcoords;
    PetscSection coordsSection;

    PetscFunctionBegin;

    PetscCall(DMGetDimension(dm, &dim));

    PetscCall(DMGetCoordinateDM(dm, &cdm));

    PetscCall(DMGetWorkArray(dm, np * dim, MPIU_SCALAR, coords));

    PetscCall(DMGetCoordinatesLocal(dm, &localCoordsVector));
    PetscCall(VecGetArray(localCoordsVector, &coordsArray));
    PetscCall(DMGetCoordinateSection(dm, &coordsSection));

    for (PetscInt i = 0; i < np; ++i) {
        PetscCall(DMPlexPointLocalRef(cdm, pArray[i], coordsArray, &vcoords));
        for (PetscInt d = 0; d < dim; ++d) {
            (*coords)[i * dim + d] = vcoords[d];
        }
    }

    PetscCall(VecRestoreArray(localCoordsVector, &coordsArray));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexVertexRestoreCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]) {
    PetscFunctionBegin;
    PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, coords));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Wrapper for the field and non-field calls. Got annoyed having this all over the place.
PetscErrorCode xDMPlexPointLocalRef(DM dm, PetscInt p, PetscInt fID, PetscScalar *array, void *ptr) {
    PetscFunctionBegin;
    if (fID >= 0)
        PetscCall(DMPlexPointLocalFieldRef(dm, p, fID, array, ptr));
    else
        PetscCall(DMPlexPointLocalRef(dm, p, array, ptr));
    PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode xDMPlexPointLocalRead(DM dm, PetscInt p, PetscInt fID, const PetscScalar *array, void *ptr) {
    PetscFunctionBegin;
    if (fID >= 0)
        PetscCall(DMPlexPointLocalFieldRead(dm, p, fID, array, ptr));
    else
        PetscCall(DMPlexPointLocalRead(dm, p, array, ptr));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * The outward facing surface area normal
 * @param dm - The DM of the data stored in vec
 * @param cell - The cell to return the outward normal to
 * @param face - Face of the cell
 * @param centroid - Centroid of the face
 * @param n - Outward facing surface area normal
 */
static PetscErrorCode DMPlexFaceCentroidOutwardAreaNormal(DM dm, PetscInt cell, PetscInt face, PetscReal *centroid, PetscReal *n) {
    PetscFunctionBegin;

    // Get the cell center
    PetscReal x0[3];
    PetscCall(DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL));

    // Centroid and normal for face/edge
    PetscReal faceArea, faceCenter[3], normal[3];
    PetscCall(DMPlexComputeCellGeometryFVM(dm, face, &faceArea, faceCenter, normal));

    PetscInt dim;
    PetscCall(DMGetDimension(dm, &dim));

    if (centroid) {
        for (PetscInt d = 0; d < dim; ++d) centroid[d] = faceCenter[d];
    }

    if (n) {
        // Make sure the normal is aligned with the vector connecting the cell center and the face center
        PetscReal sgn = 0.0;
        for (PetscInt d = 0; d < dim; ++d) {
            sgn += normal[d] * (faceCenter[d] - x0[d]);
        }

        sgn = faceArea * PetscSignReal(sgn);

        for (PetscInt d = 0; d < dim; ++d) {
            n[d] = sgn * normal[d];
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// DMPlex Notes:
// cone: Get all elements one dimension lower that use the current point
// support: Get all elements one dimension higher that use the current point
// closure: Get all elements with a lower dimension that are associated with the current point
// star: Get all elements with a higher dimension that are associated with the current point
// depth: 0 is a vertex
// height: 0 is a cell

/**
 * Given a 2D mesh compute the surface area normal in the general direction of the vector connecting the vertex to the edge center. This works for both corner normals and edge normals
 * @param dm - The mesh
 * @param vCoords - Coordinates of the vertex
 * @param pCenter - The center of either an edge or cell
 * @param nObj - The number adjacent objects to use. If pCenter refers to an edge then these should be cells. If pCenter refers to a cell then these should be edges/faces that use the vertex
 * @param objs - The points
 */
static PetscErrorCode DMPlexSurfaceAreaNormal2D_Internal(DM dm, const PetscReal vCoords[], const PetscReal pCenter[], const PetscInt nObj, const PetscInt *objs, PetscReal N[]) {
    PetscFunctionBegin;

    const PetscInt dim = 2;

    PetscReal pVertex[dim];
    for (PetscInt d = 0; d < dim; ++d) {
        // Vector from the edge center to the vertex. Used to determine the outward surface area normal
        pVertex[d] = vCoords[d] - pCenter[d];

        N[d] = 0.0;
    }

    for (PetscInt c = 0; c < nObj; ++c) {
        PetscReal objCenter[dim];  // Center of the object
        PetscReal pObj[dim];       // Vector connecting center of the object and the location pCenter
        PetscCall(DMPlexComputeCellGeometryFVM(dm, objs[c], NULL, objCenter, NULL));

        // Vector from the edge center to the cell center
        for (PetscInt d = 0; d < dim; ++d) {
            pObj[d] = objCenter[d] - pCenter[d];
        }

        // The sign of the cross-product between vectors pObj and pVertex is used to determine outward normal.
        PetscReal sgn = PetscSignReal(pObj[0] * pVertex[1] - pObj[1] * pVertex[0]);

        N[0] += sgn * pObj[1];
        N[1] -= sgn * pObj[0];
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * Given a 3D mesh compute the surface area normal in the general direction of the vector connecting the vertex to the edge center. This works for both corner normals and edge normals
 * @param dm - The mesh
 * @param targetCell - For edge normals set this to -1. For cell normals set it to the cell with the corner
 * @param vCoords - Coordinates of the vertex
 * @param pCenter - The center of either an edge or cell
 * @param nObj - The number adjacent objects to use. If pCenter refers to an edge then these should be cells. If pCenter refers to a cell then these should be edges/faces that use the vertex
 * @param objs - The points
 */
static PetscErrorCode DMPlexSurfaceAreaNormal3D_Internal(DM dm, const PetscInt targetCell, const PetscReal vCoords[], const PetscReal edgeCenter[], const PetscInt nFaces, const PetscInt *faces,
                                                         PetscReal N[]) {
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
            if (targetCell < 0 || targetCell == cells[c]) {
                PetscReal cellCenter[dim];
                PetscReal cellEdge[dim];  // Vector from cell center to edge center
                PetscReal cellFace[dim];  // Vector from cell center to face center
                PetscReal n[dim];         // Surface area normal. Needs to be corrected to align with the vector from the vertex to the edgeCenter

                PetscCall(DMPlexComputeCellGeometryFVM(dm, cells[c], NULL, cellCenter, NULL));

                for (PetscInt d = 0; d < dim; ++d) {
                    cellEdge[d] = edgeCenter[d] - cellCenter[d];
                    cellFace[d] = faceCenter[d] - cellCenter[d];
                }

                // Surface area normal. Take into account the 0.5 needed to compute the area of the triangle
                n[0] = 0.5 * (cellEdge[1] * cellFace[2] - cellEdge[2] * cellFace[1]);
                n[1] = 0.5 * (cellEdge[2] * cellFace[0] - cellEdge[0] * cellFace[2]);
                n[2] = 0.5 * (cellEdge[0] * cellFace[1] - cellEdge[1] * cellFace[0]);

                // The sign of the dot product of surface area normal and vector connecting vertex to edge center gives the correction
                PetscReal sgn = PetscSignReal(vertexEdge[0] * n[0] + vertexEdge[1] * n[1] + vertexEdge[2] * n[2]);

                for (PetscInt d = 0; d < dim; ++d) {
                    N[d] += sgn * n[d];
                }
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Compute the edge surface area normal as defined in Morgan and Waltz with respect to a given vertex and an edge center
// NOTE: This does NOT check if the vertex and edge are actually associated with each other.
static PetscErrorCode DMPlexEdgeSurfaceAreaNormal(DM dm, const PetscInt v, const PetscInt e, PetscReal N[]) {
    PetscFunctionBegin;

    PetscReal edgeCenter[3], vCoords[3];
    PetscInt dim;
    PetscInt nFace;
    const PetscInt *faces;

    PetscCall(DMGetDimension(dm, &dim));

    PetscCall(DMPlexComputeCellGeometryFVM(dm, v, NULL, vCoords, NULL));
    PetscCall(DMPlexComputeCellGeometryFVM(dm, e, NULL, edgeCenter, NULL));

    // Get all of the cells(2D) or faces(3D) associated with this edge
    PetscCall(DMPlexGetSupportSize(dm, e, &nFace));
    PetscCall(DMPlexGetSupport(dm, e, &faces));

    switch (dim) {
        case 1:
            N[0] = PetscSignReal(edgeCenter[0] - vCoords[0]);  // In 1D this is simply the vector connecting the vertex to the edge center
            break;
        case 2:
            DMPlexSurfaceAreaNormal2D_Internal(dm, vCoords, edgeCenter, nFace, faces, N);
            break;
        case 3:
            DMPlexSurfaceAreaNormal3D_Internal(dm, -1, vCoords, edgeCenter, nFace, faces, N);
            break;
        default:
            SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "DMPlexEdgeSurfaceAreaNormal can not handle dimensions of %" PetscInt_FMT, dim);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSortedArrayComplement(const PetscInt nb, const PetscInt b[], PetscInt *na, PetscInt a[]) {
    PetscFunctionBegin;

    PetscInt i = 0, j = 0;
    PetscInt n = 0;

    while (i < *na && j < nb) {
        if (a[i] < b[j]) {  // If the current element in b[] is larger, therefore a[i] cannot be included in b[j..p-1].
            a[n++] = a[i];
            i++;
        } else if (a[i] > b[j]) {  // Smaller elements of b[] are skipped
            j++;
        } else {  // Same elements detected (skipping in both arrays)
            i++;
            j++;
        }
    }

    // If a[] is larger than b[] then all remaining values must be unique as this is a sorted array.
    for (; i < *na; ++i) {
        a[n++] = a[i];
    }

    // Assign the number of unique values
    *na = n;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscSortedArrayCommon(const PetscInt na, const PetscInt a[], PetscInt *nb, PetscInt b[]) {
    PetscFunctionBegin;

    PetscInt i = 0, j = 0;
    PetscInt n = 0;

    while (i < na && j < *nb) {
        if (a[i] < b[j]) {
            i++;
        } else if (a[i] > b[j]) {
            j++;
        } else {
            b[n++] = b[j];
            i++;
            j++;
        }
    }

    *nb = n;

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Get points which are common between two DMPlex points. For example, if p1 is a cell and p2 is a vertex on the cell with depth=1 this will
//   return the edges common to both p1 and p2
PetscErrorCode DMPlexGetCommonPoints(DM dm, const PetscInt p1, const PetscInt p2, const PetscInt depth, PetscInt *nPoints, PetscInt *points[]) {
    PetscInt pStart, pEnd;
    PetscInt pts[2] = {p1, p2};
    PetscInt inputDepths[2] = {-1, -1};
    PetscInt dim;
    PetscInt nList[2] = {0, 0}, list[2][100];  // 100 should be much larger than the maximum number of elements using a point.

    PetscFunctionBegin;

    PetscCall(DMGetDimension(dm, &dim));

    PetscCall(DMPlexGetDepthStratum(dm, depth, &pStart, &pEnd));  // Range of points to search for

    // Get the depths of p1 and p2
    PetscCall(DMPlexGetPointDepth(dm, p1, &inputDepths[0]));
    PetscCall(DMPlexGetPointDepth(dm, p2, &inputDepths[1]));

    for (PetscInt i = 0; i < 2; ++i) {
        PetscInt nClosure;
        static PetscInt *closure;

        PetscCall(DMPlexGetTransitiveClosure(dm, pts[i], PetscBool(depth < inputDepths[i]), &nClosure, &closure));

        for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
            if (closure[cl] >= pStart && closure[cl] < pEnd) {
                list[i][nList[i]++] = closure[cl];
            }
        }

        PetscCheck(nList[i] < 101, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of elements using point %" PetscInt_FMT " exceeds 100.", pts[i]);

        PetscCall(DMPlexRestoreTransitiveClosure(dm, pts[i], PetscBool(depth < inputDepths[i]), &nClosure, &closure));

        PetscCall(PetscSortInt(nList[i], list[i]));
    }

    PetscCall(PetscSortedArrayCommon(nList[0], list[0], &nList[1], list[1]));

    *nPoints = nList[1];

    if (*nPoints > 0) {
        PetscCall(DMGetWorkArray(dm, *nPoints, MPIU_INT, points));
        PetscCall(PetscArraycpy(*points, list[1], *nPoints));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexRestoreCommonPoints(DM dm, const PetscInt p1, const PetscInt p2, const PetscInt depth, PetscInt *nPoints, PetscInt *points[]) {
    PetscFunctionBegin;

    PetscCall(DMRestoreWorkArray(dm, *nPoints, MPIU_INT, points));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCornerSurfaceAreaNormal(DM dm, const PetscInt v, const PetscInt c, PetscReal N[]) {
    PetscFunctionBegin;

    PetscReal vCoords[3];
    PetscInt dim;
    PetscInt nEdges, *edges;

    PetscCall(DMGetDimension(dm, &dim));

    PetscCall(DMPlexComputeCellGeometryFVM(dm, v, NULL, vCoords, NULL));

    // Get all edges in the cell that use the vertex
    PetscCall(DMPlexGetCommonPoints(dm, v, c, 1, &nEdges, &edges));

    for (PetscInt d = 0; d < dim; ++d) N[d] = 0.0;

    for (PetscInt e = 0; e < nEdges; ++e) {
        PetscReal edgeCenter[3] = {0.0, 0.0, 0.0}, n[3] = {0.0, 0.0, 0.0};

        PetscCall(DMPlexComputeCellGeometryFVM(dm, edges[e], NULL, edgeCenter, NULL));

        switch (dim) {
            case 1: {
                n[0] = PetscSignReal(edgeCenter[0] - vCoords[0]);  // In 1D this is simply the vector connecting the vertex to the edge center
            } break;
            case 2: {
                PetscCall(DMPlexSurfaceAreaNormal2D_Internal(dm, vCoords, edgeCenter, 1, &c, n));
            } break;
            case 3: {
                // Get all of the cells associated with this edges and this cell
                PetscInt nFaces = 0, *faces = nullptr;
                PetscCall(DMPlexGetCommonPoints(dm, edges[e], c, 2, &nFaces, &faces));
                PetscCall(DMPlexSurfaceAreaNormal3D_Internal(dm, c, vCoords, edgeCenter, nFaces, faces, n));
                PetscCall(DMPlexRestoreCommonPoints(dm, edges[e], c, 2, &nFaces, &faces));
            } break;
            default:
                SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "DMPlexEdgeSurfaceAreaNormal can not handle dimensions of %" PetscInt_FMT, dim);
        }

        for (PetscInt d = 0; d < dim; ++d) N[d] += n[d];
    }

    PetscCall(DMPlexRestoreCommonPoints(dm, v, c, 1, &nEdges, &edges));

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

    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star));  // Everything using this vertex

    *vol = 0.0;

    for (PetscInt st = 0; st < nStar * 2; st += 2) {
        if (star[st] >= cStart && star[st] < cEnd) {
            PetscReal cellVol;
            PetscInt nCorners;

            PetscCall(DMPlexComputeCellGeometryFVM(dm, star[st], &cellVol, NULL, NULL));

            // The number of corners a cell can be divided into equals the number of vertices associated with that cell
            PetscCall(DMPlexCellGetNumVertices(dm, star[st], &nCorners));

            *vol += cellVol / nCorners;
        }
    }

    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// NOTE: I am not sure how this will handle corner of a domain.

// Compute the finite-difference derivative approximation using the Eq. (11) from "3D level set methods for evolving fronts on tetrahedral
//    meshes with adaptive mesh refinement", by Morgan and Waltz, JCP 336 (2017) 492-512.
//   This should be second-order accurate for both triangles and quads
PetscErrorCode DMPlexVertexGradFromVertex(DM dm, const PetscInt v, Vec data, PetscInt fID, PetscInt offset, PetscScalar g[]) {
    PetscFunctionBegin;

    PetscInt nEdge;
    const PetscInt *edge;
    const PetscScalar *dataArray;
    PetscInt vStart, vEnd;
    PetscInt dim;

    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));  // Range of vertices
    PetscCheck(v >= vStart && v < vEnd, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "DMPlexVertexGradFromVertex must have a valid vertex as input.");

    PetscCall(DMGetDimension(dm, &dim));

    PetscCheck(dim > 0 && dim < 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "DMPlexVertexGradFromVertex does not support a DM of dimension %" PetscInt_FMT, dim);

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

        edgeVal = 0.5 * val[offset];

        PetscCall(xDMPlexPointLocalRead(dm, verts[1], fID, dataArray, &val));
        edgeVal += 0.5 * val[offset];

        for (PetscInt d = 0; d < dim; ++d) {
            g[d] += edgeVal * N[d];
        }
    }

    PetscCall(VecRestoreArrayRead(data, &dataArray));

    PetscReal cvVol;
    PetscCall(DMPlexVertexControlVolume(dm, v, &cvVol));
    for (PetscInt d = 0; d < dim; ++d) {
        g[d] /= cvVol;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexVertexGradFromCell(DM dm, const PetscInt v, Vec data, PetscInt fID, PetscInt offset, PetscScalar g[]) {
    PetscFunctionBegin;

    const PetscScalar *dataArray;
    PetscInt vStart, vEnd;
    PetscInt cStart, cEnd;
    PetscInt dim;
    PetscInt nStar, *star = NULL;

    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));  // Range of vertices
    PetscCheck(v >= vStart && v < vEnd, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "DMPlexVertexGradFromVertex must have a valid vertex as input.");

    PetscCall(DMGetDimension(dm, &dim));

    PetscCheck(dim > 0 && dim < 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "DMPlexVertexGradFromVertex does not support a DM of dimension %" PetscInt_FMT, dim);

    for (PetscInt d = 0; d < dim; ++d) {
        g[d] = 0.0;
    }

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // Range of cells
    PetscCall(VecGetArrayRead(data, &dataArray));

    // Everything using this vertex
    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star));
    for (PetscInt st = 0; st < nStar * 2; st += 2) {
        if (star[st] >= cStart && star[st] < cEnd) {  // It's a cell

            // Surface area normal
            PetscScalar N[3];
            PetscCall(DMPlexCornerSurfaceAreaNormal(dm, v, star[st], N));

            const PetscScalar *val;
            PetscCall(xDMPlexPointLocalRead(dm, star[st], fID, dataArray, &val));

            for (PetscInt d = 0; d < dim; ++d) {
                g[d] += val[offset] * N[d];
            }
        }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &nStar, &star));

    PetscCall(VecRestoreArrayRead(data, &dataArray));

    PetscReal cvVol;
    PetscCall(DMPlexVertexControlVolume(dm, v, &cvVol));

    for (PetscInt d = 0; d < dim; ++d) {
        g[d] /= cvVol;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCellGradFromVertex(DM dm, const PetscInt c, Vec data, PetscInt fID, PetscInt offset, PetscScalar g[]) {
    PetscFunctionBegin;

    PetscInt nFace;
    const PetscInt *faces;
    const PetscScalar *dataArray;
    PetscInt cStart, cEnd;
    PetscInt vStart, vEnd;
    PetscInt dim;

    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));   // Range of vertices
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // Range of cells
    PetscCheck(c >= cStart && c < cEnd, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "DMPlexCellToCellGrad must have a valid cell as input.");

    PetscCall(DMGetDimension(dm, &dim));

    PetscCheck(dim > 0 && dim < 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "DMPlexCellToCellGrad does not support a DM of dimension %" PetscInt_FMT, dim);

    for (PetscInt d = 0; d < dim; ++d) {
        g[d] = 0.0;
    }

    PetscCall(VecGetArrayRead(data, &dataArray));

    // Get all faces associated with the cell
    PetscCall(DMPlexGetConeSize(dm, c, &nFace));
    PetscCall(DMPlexGetCone(dm, c, &faces));
    for (PetscInt f = 0; f < nFace; ++f) {
        PetscReal N[3] = {0.0, 0.0, 0.0};
        PetscCall(DMPlexFaceCentroidOutwardAreaNormal(dm, c, faces[f], NULL, N));

        // All points associated with this face
        PetscInt nClosure, *closure = NULL;
        PetscCall(DMPlexGetTransitiveClosure(dm, faces[f], PETSC_TRUE, &nClosure, &closure));

        PetscReal cnt = 0.0, ave = 0.0;
        for (PetscInt cl = 0; cl < nClosure * 2; cl += 2) {
            if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex
                const PetscScalar *val;
                PetscCall(xDMPlexPointLocalRead(dm, closure[cl], fID, dataArray, &val));
                ave += val[offset];
                cnt += 1.0;
            }
        }

        PetscCall(DMPlexRestoreTransitiveClosure(dm, faces[f], PETSC_TRUE, &nClosure, &closure));  // Restore the points

        // Function value at the face center
        ave /= cnt;
        for (PetscInt d = 0; d < dim; ++d) {
            g[d] += ave * N[d];
        }
    }

    PetscCall(VecRestoreArrayRead(data, &dataArray));

    PetscReal vol;
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL));
    for (PetscInt d = 0; d < dim; ++d) {
        g[d] /= vol;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Copied from petsc/src/dm/impls/plex/plexgeometry
static PetscErrorCode DMPlexGetLineIntersection_2D_Internal(const PetscReal segmentA[], const PetscReal segmentB[], PetscReal intersection[], PetscBool *hasIntersection) {
    const PetscReal p0_x = segmentA[0 * 2 + 0];
    const PetscReal p0_y = segmentA[0 * 2 + 1];
    const PetscReal p1_x = segmentA[1 * 2 + 0];
    const PetscReal p1_y = segmentA[1 * 2 + 1];
    const PetscReal p2_x = segmentB[0 * 2 + 0];
    const PetscReal p2_y = segmentB[0 * 2 + 1];
    const PetscReal p3_x = segmentB[1 * 2 + 0];
    const PetscReal p3_y = segmentB[1 * 2 + 1];
    const PetscReal s1_x = p1_x - p0_x;
    const PetscReal s1_y = p1_y - p0_y;
    const PetscReal s2_x = p3_x - p2_x;
    const PetscReal s2_y = p3_y - p2_y;
    const PetscReal denom = (-s2_x * s1_y + s1_x * s2_y);

    PetscFunctionBegin;
    *hasIntersection = PETSC_FALSE;
    /* Non-parallel lines */
    if (denom != 0.0) {
        const PetscReal s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / denom;
        const PetscReal t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / denom;

        if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
            *hasIntersection = PETSC_TRUE;
            if (intersection) {
                intersection[0] = p0_x + (t * s1_x);
                intersection[1] = p0_y + (t * s1_y);
            }
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* The plane is segmentB x segmentC: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection */
static PetscErrorCode DMPlexGetLineTriangleIntersection_3D_Internal(const PetscReal segmentA[], const PetscReal segmentB[], const PetscReal segmentC[], PetscReal intersection[],
                                                                    PetscBool *hasIntersection) {
    const PetscReal p0_x = segmentA[0 * 3 + 0];
    const PetscReal p0_y = segmentA[0 * 3 + 1];
    const PetscReal p0_z = segmentA[0 * 3 + 2];
    const PetscReal p1_x = segmentA[1 * 3 + 0];
    const PetscReal p1_y = segmentA[1 * 3 + 1];
    const PetscReal p1_z = segmentA[1 * 3 + 2];

    const PetscReal q0_x = segmentB[0 * 3 + 0];
    const PetscReal q0_y = segmentB[0 * 3 + 1];
    const PetscReal q0_z = segmentB[0 * 3 + 2];
    const PetscReal q1_x = segmentB[1 * 3 + 0];
    const PetscReal q1_y = segmentB[1 * 3 + 1];
    const PetscReal q1_z = segmentB[1 * 3 + 2];

    const PetscReal r0_x = segmentC[0 * 3 + 0];
    const PetscReal r0_y = segmentC[0 * 3 + 1];
    const PetscReal r0_z = segmentC[0 * 3 + 2];
    const PetscReal r1_x = segmentC[1 * 3 + 0];
    const PetscReal r1_y = segmentC[1 * 3 + 1];
    const PetscReal r1_z = segmentC[1 * 3 + 2];

    const PetscReal s0_x = p1_x - p0_x;
    const PetscReal s0_y = p1_y - p0_y;
    const PetscReal s0_z = p1_z - p0_z;
    const PetscReal s1_x = q1_x - q0_x;
    const PetscReal s1_y = q1_y - q0_y;
    const PetscReal s1_z = q1_z - q0_z;
    const PetscReal s2_x = r1_x - r0_x;
    const PetscReal s2_y = r1_y - r0_y;
    const PetscReal s2_z = r1_z - r0_z;
    const PetscReal s3_x = s1_y * s2_z - s1_z * s2_y; /* s1 x s2 */
    const PetscReal s3_y = s1_z * s2_x - s1_x * s2_z;
    const PetscReal s3_z = s1_x * s2_y - s1_y * s2_x;
    const PetscReal s4_x = s0_y * s2_z - s0_z * s2_y; /* s0 x s2 */
    const PetscReal s4_y = s0_z * s2_x - s0_x * s2_z;
    const PetscReal s4_z = s0_x * s2_y - s0_y * s2_x;
    const PetscReal s5_x = s1_y * s0_z - s1_z * s0_y; /* s1 x s0 */
    const PetscReal s5_y = s1_z * s0_x - s1_x * s0_z;
    const PetscReal s5_z = s1_x * s0_y - s1_y * s0_x;
    const PetscReal denom = -(s0_x * s3_x + s0_y * s3_y + s0_z * s3_z); /* -s0 . (s1 x s2) */

    PetscFunctionBegin;
    *hasIntersection = PETSC_FALSE;
    /* Line not parallel to plane */
    if (denom != 0.0) {
        const PetscReal t = (s3_x * (p0_x - q0_x) + s3_y * (p0_y - q0_y) + s3_z * (p0_z - q0_z)) / denom;
        const PetscReal u = (s4_x * (p0_x - q0_x) + s4_y * (p0_y - q0_y) + s4_z * (p0_z - q0_z)) / denom;
        const PetscReal v = (s5_x * (p0_x - q0_x) + s5_y * (p0_y - q0_y) + s5_z * (p0_z - q0_z)) / denom;

        if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
            *hasIntersection = PETSC_TRUE;
            if (intersection) {
                intersection[0] = p0_x + (t * s0_x);
                intersection[1] = p0_y + (t * s0_y);
                intersection[2] = p0_z + (t * s0_z);
            }
        }
        // printf("%+f\t%+f\t%+f\n", t, u, v);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Determine the intersection of a face and a line
PetscErrorCode DMPlexFaceLineIntersection(DM dm, const PetscInt f, const PetscReal segment[], PetscReal intersection[], PetscBool *hasIntersection) {
    PetscInt dim;
    const PetscScalar *array;
    PetscScalar *coords = NULL;
    PetscInt numCoords;
    PetscBool isDG;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));

    PetscCall(DMPlexGetCellCoordinates(dm, f, &isDG, &numCoords, &array, &coords));

    switch (dim) {
        case 1: {
            *hasIntersection = PETSC_FALSE;
        } break;
        case 2: {
            PetscCall(DMPlexGetLineIntersection_2D_Internal(coords, segment, intersection, hasIntersection));
        } break;
        case 3: {
            PetscReal segmentA[6], segmentB[6];

            // Triangle 0 - 1 - 2
            segmentA[0] = segmentB[0] = coords[3];
            segmentA[1] = segmentB[1] = coords[4];
            segmentA[2] = segmentB[2] = coords[5];

            segmentA[3] = coords[0];
            segmentA[4] = coords[1];
            segmentA[5] = coords[2];

            segmentB[3] = coords[6];
            segmentB[4] = coords[7];
            segmentB[5] = coords[8];

            PetscCall(DMPlexGetLineTriangleIntersection_3D_Internal(segment, segmentA, segmentB, intersection, hasIntersection));

            if (numCoords == 12 && !(*hasIntersection)) {  // The face is a quad, so try triangle 0 - 3 - 2
                segmentA[0] = segmentB[0] = coords[9];
                segmentA[1] = segmentB[1] = coords[10];
                segmentA[2] = segmentB[2] = coords[11];
                PetscCall(DMPlexGetLineTriangleIntersection_3D_Internal(segment, segmentA, segmentB, intersection, hasIntersection));
            }
        } break;
        default:
            PetscFunctionReturn(PETSC_ERR_SUP);
    }

    PetscCall(DMPlexRestoreCellCoordinates(dm, f, &isDG, &numCoords, &array, &coords));

    PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/hashmapi.h>
PetscErrorCode DMPlexCellGradFromCell(DM dm, const PetscInt c, Vec data, PetscInt fID, PetscInt offset, PetscScalar g[]) {
    PetscFunctionBegin;

    PetscInt cStart, cEnd;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // Range of cells
    PetscCheck(c >= cStart && c < cEnd, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "DMPlexCellToCellGrad must have a valid cell as input.");

    PetscInt dim;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCheck(dim > 1 && dim < 4, PETSC_COMM_SELF, PETSC_ERR_SUP, "DMPlexCellToCellGrad does not support a DM of dimension %" PetscInt_FMT, dim);

    const PetscScalar *dataArray;
    PetscCall(VecGetArrayRead(data, &dataArray));

    // Get all vertices of the cell
    PetscInt nVert, *verts;
    PetscCall(DMPlexCellGetVertices(dm, c, &nVert, &verts));

    PetscHMapI hash = NULL;  // Used to convert from vertex numbering to 0->nVert-1
    PetscCall(PetscHMapICreate(&hash));

    PetscReal *vertVals;
    PetscCall(DMGetWorkArray(dm, nVert, MPIU_REAL, &vertVals));

    // Locations of the vertices
    PetscReal *vertCoords;
    PetscCall(DMPlexVertexGetCoordinates(dm, nVert, verts, &vertCoords));
    for (PetscInt v = 0; v < nVert; ++v) {
        PetscCall(PetscHMapISet(hash, verts[v], v));

        vertVals[v] = 0.0;

        PetscInt nCells, *cells;
        PetscCall(DMPlexVertexGetCells(dm, verts[v], &nCells, &cells));

        PetscReal totalWt = 0.0;
        for (PetscInt i = 0; i < nCells; ++i) {
            PetscReal *cellVal, cellCenter[3], wt = 0.0;
            PetscCall(xDMPlexPointLocalRead(dm, cells[i], fID, dataArray, &cellVal));
            PetscCall(DMPlexComputeCellGeometryFVM(dm, cells[i], NULL, cellCenter, NULL));
            for (PetscInt d = 0; d < dim; ++d) wt += PetscSqr(cellCenter[d] - vertCoords[v * dim + d]);
            wt = 1.0 / PetscSqrtReal(wt);
            totalWt += wt;
            vertVals[v] += wt * cellVal[offset];
        }
        PetscCall(DMPlexVertexRestoreCells(dm, verts[v], &nCells, &cells));

        vertVals[v] /= totalWt;
    }
    PetscCall(DMPlexVertexRestoreCoordinates(dm, nVert, verts, &vertCoords));
    PetscCall(DMPlexCellRestoreVertices(dm, c, &nVert, &verts));

    for (PetscInt d = 0; d < dim; ++d) g[d] = 0.0;

    PetscInt nFaces;
    const PetscInt *faces;
    PetscCall(DMPlexGetConeSize(dm, c, &nFaces));
    PetscCall(DMPlexGetCone(dm, c, &faces));
    for (PetscInt f = 0; f < nFaces; ++f) {
        PetscInt nVert, *verts;
        PetscCall(DMPlexCellGetVertices(dm, faces[f], &nVert, &verts));

        PetscReal faceValue = 0.0;
        for (PetscInt v = 0; v < nVert; ++v) {
            PetscInt id;
            PetscHMapIGet(hash, verts[v], &id);
            faceValue += vertVals[id];
        }
        faceValue /= nVert;

        PetscCall(DMPlexCellRestoreVertices(dm, faces[f], &nVert, &verts));

        PetscReal N[dim];
        PetscCall(DMPlexFaceCentroidOutwardAreaNormal(dm, c, faces[f], NULL, N));

        for (PetscInt d = 0; d < dim; ++d) g[d] += faceValue * N[d];
    }

    PetscHMapIDestroy(&hash);
    PetscCall(DMRestoreWorkArray(dm, nVert, MPIU_REAL, &vertVals));

    // Center of the cell
    PetscReal cellVolume;
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &cellVolume, NULL, NULL));
    for (PetscInt d = 0; d < dim; ++d) g[d] /= cellVolume;

    PetscFunctionReturn(PETSC_SUCCESS);
}

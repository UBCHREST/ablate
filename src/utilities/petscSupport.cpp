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
 */
PetscErrorCode DMPlexGetContainingCell(DM dm, PetscScalar *xyz, PetscInt *cell) {
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

    PetscFunctionReturn(0);
}

/**
 * Return all cells which share an vertex or edge/face with a center cell
 * Inputs:
 *    dm - The mesh
 *    x0 - The location of the true center cell
 *    p - The cell to get the neighboors of
 *    maxDist - Maximum distance from p to consider adding
 *    useCells - Should we include cells which share a vertex (FALSE) or an edge/face (TRUE)
 *
 * Outputs:
 *    nCells - Number of cells found
 *    cells - The IDs of the cells found.
 */
static PetscErrorCode DMPlexGetNeighborCells_Internal(DM dm, PetscReal x0[3], PetscInt p, PetscReal maxDist, PetscBool useCells, PetscInt *nCells, PetscInt *cells[]) {
    PetscInt cStart, cEnd, vStart, vEnd;
    PetscInt cl, nClosure, *closure = NULL;
    PetscInt st, nStar, *star = NULL;
    PetscInt n, list[10000];  // To avoid memory reallocation just make the list bigger than it would ever need to be. Will look at doing something else in the future.
    PetscInt i, dim;
    PetscReal x[3], dist;

    PetscFunctionBegin;

    PetscCall(DMGetDimension(dm, &dim));  // The dimension of the grid

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));  // Range of cells

    if (useCells) {
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

    PetscFunctionReturn(0);
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

    PetscFunctionReturn(0);
}

/**
 * Return all values in sorted array a that are NOT in sorted array b. This is done in-place on array a.
 * Inputs:
 *    nb - Size of sorted array b[]
 *    b - Array of integers
 *    na - Size of sorted array a[]
 *    a - Array or integers
 *
 * Outputs:
 *    na - Number of integers in b but not in a
 *    a - All integers in b but not in a
 */
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
        } else if (a[i] == b[j]) {  // Same elements detected (skipping in both arrays)
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

    PetscFunctionReturn(0);
}

/**
 * Return the list of neighboring cells/vertices to cell p using a combination of number of levels and maximum distance
 * dm - The mesh
 * maxLevels - Number of neighboring cells/vertices to check
 * maxDist - Maximum distance to include
 * numberCells - The number of cells/vertices to return.
 * nCells - Number of neighboring cells/vertices
 * cells - The list of neighboring cell/vertices IDs
 *
 * Note: The intended use is to use either maxLevels OR maxDist OR minNumberCells. Right now a check isn't done on only selecting one, but that might be added in the future.
 */
PetscErrorCode DMPlexGetNeighbors(DM dm, PetscInt p, PetscInt maxLevels, PetscReal maxDist, PetscInt numberCells, PetscBool useCells, PetscBool returnNeighborVertices, PetscInt *nCells,
                                  PetscInt *cells[]) {
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
        for (PetscInt i = 0; i < closureSize * 2; i += 2) {
            PetscInt point = closure[i];
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
    PetscCall(PetscMalloc1(numberCells, cells));
    PetscCall(PetscArraycpy(*cells, list, numberCells));
    PetscCall(PetscIntSortSemiOrdered(numberCells, *cells));
    *nCells = numberCells;

    PetscFunctionReturn(0);
}

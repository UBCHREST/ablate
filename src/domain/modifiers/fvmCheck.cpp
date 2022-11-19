#include "fvmCheck.hpp"
#include <petsc/private/sectionimpl.h>
#include <array>
#include <iomanip>
#include <map>
#include <set>
#include <utility>
#include "utilities/mathUtilities.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::domain::modifiers::FvmCheck::FvmCheck(std::shared_ptr<domain::Region> fvmRegion, int expectedFaceCount, int expectedNodeCount)
    : region(std::move(fvmRegion)), expectedFaceCount(expectedFaceCount), expectedNodeCount(expectedNodeCount) {}

void ablate::domain::modifiers::FvmCheck::Modify(DM& dm) {
    PetscInt depth;
    DMPlexGetDepth(dm, &depth) >> checkError;

    // Get the faces in this range
    ablate::solver::Range faceRange;
    GetRange(dm, depth - 1, faceRange);

    // get the label for this region
    DMLabel regionLabel = nullptr;
    PetscInt regionValue = 0;
    domain::Region::GetLabel(region, dm, regionLabel, regionValue);

    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

    // compute the dm geometry
    Vec cellGeomVec, faceGeomVec;
    DMPlexComputeGeometryFVM(dm, &cellGeomVec, &faceGeomVec) >> checkError;

    // Get the dim
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;

    // Get the dm for each value
    DM cellDM, faceDM;
    VecGetDM(cellGeomVec, &cellDM) >> checkError;
    VecGetDM(faceGeomVec, &faceDM) >> checkError;

    // Get the array data from the geom vec
    const PetscScalar* cellGeomArray;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    // store a map of summed area
    std::map<PetscInt, std::array<PetscReal, 3>> cellAreas;

    // Store a count for the number of faces on each cell
    std::map<PetscInt, PetscInt> faceCount;

    // check if it is an exterior boundary cell ghost
    PetscInt boundaryCellStart;
    DMPlexGetGhostCellStratum(dm, &boundaryCellStart, nullptr) >> checkError;

    if (boundaryCellStart < 0 && region == nullptr) {
        throw std::invalid_argument(
            "The FVM check cannot be used over the entire mesh if there are no boundary ghost cells. Add boundary ghost cells with ablate::domain::modifiers::GhostBoundaryCells");
    }

    // Get the fv geom to find the smallest cell in mesh
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> checkError;
    std::cout << "The minimum cell length is: " << minCellRadius << std::endl;

    // March over each face in this region
    for (PetscInt f = faceRange.start; f < faceRange.end; ++f) {
        const PetscInt face = faceRange.points ? faceRange.points[f] : f;

        // make sure that this is a valid face
        PetscInt ghost = -1, nsupp, nchild;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, face, &ghost) >> checkError;
        }
        DMPlexGetSupportSize(dm, face, &nsupp) >> checkError;
        DMPlexGetTreeChildren(dm, face, &nchild, nullptr) >> checkError;
        if (ghost >= 0 || nsupp != 2 || nchild > 0) continue;

        // Get the face geometry
        const PetscInt* faceCells;
        PetscFVFaceGeom* fg;
        PetscFVCellGeom *cgL, *cgR;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;
        DMPlexGetSupport(dm, face, &faceCells) >> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[0], cellGeomArray, &cgL) >> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[1], cellGeomArray, &cgR) >> checkError;

        PetscInt leftFlowLabelValue = regionValue;
        PetscInt rightFlowLabelValue = regionValue;
        if (regionLabel) {
            DMLabelGetValue(regionLabel, faceCells[0], &leftFlowLabelValue);
            DMLabelGetValue(regionLabel, faceCells[1], &rightFlowLabelValue);
        }

        // Check the normal direction, it should go from left[0] to right[1]
        PetscScalar lToR[3] = {0.0, 0.0, 0.0};
        ablate::utilities::MathUtilities::Subtract(dim, cgR->centroid, cgL->centroid, lToR);

        // Check the normal direction
        auto direction = ablate::utilities::MathUtilities::DotVector(dim, lToR, fg->normal);
        if (direction <= 0) {
            std::cout << "Normal in wrong direction for face: " << face << " with norm [" << fg->normal[0] << ", " << fg->normal[1] << ", " << fg->normal[2] << "]" << std::endl;
            std::cout << "\t leftCell " << faceCells[0] << ": [" << cgL->centroid[0] << ", " << cgL->centroid[1] << ", " << cgL->centroid[2] << "]" << std::endl;
            std::cout << "\t rightCell " << faceCells[1] << ": [" << cgR->centroid[0] << ", " << cgR->centroid[1] << ", " << cgR->centroid[2] << "]" << std::endl;
        }

        // check value
        PetscInt cellLabelValue = regionValue;
        ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> checkError;
        }
        if (regionLabel) {
            DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> checkError;
        }
        if (ghost <= 0 && regionValue == cellLabelValue && (boundaryCellStart < 0 || faceCells[0] < boundaryCellStart)) {
            if (!cellAreas.count(faceCells[0])) {
                cellAreas[faceCells[0]] = {0, 0, 0};
                faceCount[faceCells[0]] = 0;
            }

            cellAreas[faceCells[0]][0] -= fg->normal[0];
            cellAreas[faceCells[0]][1] -= fg->normal[1];
            cellAreas[faceCells[0]][2] -= fg->normal[2];
            faceCount[faceCells[0]]++;
        }

        cellLabelValue = regionValue;
        ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> checkError;
        }
        if (regionLabel) {
            DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> checkError;
        }
        if (ghost <= 0 && regionValue == cellLabelValue && (boundaryCellStart < 0 || faceCells[1] < boundaryCellStart)) {
            if (!cellAreas.count(faceCells[1])) {
                cellAreas[faceCells[1]] = {0, 0, 0};
                faceCount[faceCells[1]] = 0;
            }

            cellAreas[faceCells[1]][0] += fg->normal[0];
            cellAreas[faceCells[1]][1] += fg->normal[1];
            cellAreas[faceCells[1]][2] += fg->normal[2];
            faceCount[faceCells[1]]++;
        }
    }

    utilities::MpiUtilities::RoundRobin(PetscObjectComm((PetscObject)dm), [&](int rank) {
        // Check over each cell that was contributed to
        for (const auto& cellInfo : cellAreas) {
            bool nonZeroArea = false;
            for (PetscInt d = 0; d < dim; d++) {
                // Make sure each area contribution is zero
                if (cellInfo.second[d] > 1E-12 || cellInfo.second[d] < -1E-12) {
                    nonZeroArea = true;
                }
            }

            // check the expected faces
            bool wrongFaceCount = expectedFaceCount && (faceCount[cellInfo.first] != expectedFaceCount);

            bool wrongNodeCount = false;
            std::set<PetscInt> nodesInCell;
            if (expectedNodeCount) {
                // Count the number of nodes in this cell
                PetscInt* points = nullptr;
                PetscInt numPoints;
                DMPlexGetTransitiveClosure(dm, cellInfo.first, PETSC_TRUE, &numPoints, &points) >> checkError;

                for (PetscInt p = 0; p < numPoints; p++) {
                    PetscInt point = points[p * 2];

                    // Check the depth
                    PetscInt pointDepth = 0;
                    DMPlexGetPointDepth(dm, point, &pointDepth) >> checkError;

                    // check if node
                    if (pointDepth == 0) {
                        nodesInCell.insert(point);
                    }
                }

                wrongNodeCount = (expectedNodeCount != (PetscInt)nodesInCell.size());
            }

            if (nonZeroArea || wrongFaceCount || wrongNodeCount) {
                PetscFVCellGeom* cg;
                DMPlexPointLocalRead(cellDM, cellInfo.first, cellGeomArray, &cg) >> checkError;

                std::cout << "Issues with Cell: " << cellInfo.first << " at [" << cg->centroid[0] << ", " << cg->centroid[1] << ", " << cg->centroid[2] << "]" << std::endl;
                if (expectedFaceCount) {
                    std::cout << "wrongFaceCount: " << wrongFaceCount << std::endl;
                    std::cout << "\tfaceCount: " << faceCount[cellInfo.first] << std::endl;
                }
                if (expectedNodeCount) {
                    std::cout << "wrongNodeCount: " << wrongNodeCount << std::endl;
                    std::cout << "\tnodeCount: " << nodesInCell.size() << std::endl;
                }
                std::cout << "nonZeroArea: " << nonZeroArea << std::endl;
                for (PetscInt d = 0; d < dim; d++) {
                    std::cout << "\tarea[" << d << "]: " << std::setprecision(16) << cellInfo.second[d] << std::endl;
                }

                // DM dm, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[]
                PetscReal volume;
                DMPlexComputeCellGeometryFVM(dm, cellInfo.first, &volume, nullptr, nullptr) >> checkError;
                std::cout << "volume: " << std::setprecision(16) << volume << std::endl;

                // Print all labels at this cell
                // output the labels at this point
                PetscInt numberLabels;
                DMGetNumLabels(dm, &numberLabels);
                std::cout << "\tLabels: ";
                for (PetscInt l = 0; l < numberLabels; l++) {
                    DMLabel labelCheck;
                    DMGetLabelByNum(dm, l, &labelCheck) >> checkError;
                    const char* labelName;
                    DMGetLabelName(dm, l, &labelName) >> checkError;
                    PetscInt labelCheckValue;
                    DMLabelGetValue(labelCheck, cellInfo.first, &labelCheckValue) >> checkError;
                    PetscInt labelDefaultValue;
                    DMLabelGetDefaultValue(labelCheck, &labelDefaultValue) >> checkError;
                    if (labelDefaultValue != labelCheckValue) {
                        std::cout << labelName << "(" << labelCheckValue << ") ";
                    }
                }
                std::cout << std::endl;

                // March over each face connected to this cell
                const PetscInt* faces;
                PetscInt numberFaces;
                DMPlexGetConeSize(dm, cellInfo.first, &numberFaces) >> checkError;
                DMPlexGetCone(dm, cellInfo.first, &faces) >> checkError;
                std::cout << "faces: " << std::endl;

                for (PetscInt f = 0; f < numberFaces; f++) {
                    PetscInt face = faces[f];

                    // get the cells that touch this face
                    const PetscInt* cells;
                    PetscInt numberCells;
                    DMPlexGetSupport(dm, face, &cells) >> checkError;
                    DMPlexGetSupportSize(dm, face, &numberCells) >> checkError;

                    std::cout << "\tface: " << face << std::endl;

                    // compute the area for this face
                    PetscReal area;
                    PetscReal normal[3];
                    PetscReal centroid[3];

                    // DM dm, PetscInt cell, PetscReal *vol, PetscReal centroid[], PetscReal normal[]
                    DMPlexComputeCellGeometryFVM(dm, face, &area, centroid, normal) >> checkError;

                    // check to see if we need to flip the area
                    PetscInt leftCell = cells[0];
                    PetscInt rightCell = cells[1];
                    PetscScalar leftCellCentroid[3];
                    if (leftCell >= boundaryCellStart) {
                        PetscArraycpy(leftCellCentroid, centroid, dim);
                    } else {
                        DMPlexComputeCellGeometryFVM(dm, leftCell, nullptr, leftCellCentroid, nullptr) >> checkError;
                    }
                    PetscScalar rightCellCentroid[3];
                    if (numberCells < 2 || rightCell >= boundaryCellStart) {
                        PetscArraycpy(rightCellCentroid, centroid, dim);

                    } else {
                        DMPlexComputeCellGeometryFVM(dm, rightCell, nullptr, rightCellCentroid, nullptr) >> checkError;
                    }

                    // Check the normal direction, it should go from left[0] to right[1]
                    PetscScalar lToR[3] = {0.0, 0.0, 0.0};
                    ablate::utilities::MathUtilities::Subtract(dim, rightCellCentroid, leftCellCentroid, lToR);

                    // Check if left or right
                    PetscInt leftOrRight = leftCell == cellInfo.first ? -1 : 1;

                    // Check the normal direction
                    auto direction = PetscSignReal(ablate::utilities::MathUtilities::DotVector(dim, lToR, normal));

                    std::cout << "\t\torgNormal: [" << normal[0] << ", " << normal[1] << ", " << normal[2] << "]" << std::endl;
                    std::cout << "\t\tleftToRight: [" << lToR[0] << ", " << lToR[1] << ", " << lToR[2] << "]" << std::endl;
                    std::cout << "\t\tleftToRightDotNormal: " << ablate::utilities::MathUtilities::DotVector(dim, lToR, normal) << std::endl;
                    std::cout << "\t\tleftCellCentroid: [" << leftCellCentroid[0] << ", " << leftCellCentroid[1] << ", " << leftCellCentroid[2] << "]" << std::endl;
                    std::cout << "\t\trightCellCentroid: [" << rightCellCentroid[0] << ", " << rightCellCentroid[1] << ", " << rightCellCentroid[2] << "]" << std::endl;
                    std::cout << "\t\tfaceCentroid: [" << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << "]" << std::endl;

                    // Scale the normal by the area and direction
                    for (PetscInt d = 0; d < dim; d++) {
                        normal[d] *= area * direction;
                    }

                    std::cout << "\t\tleft(-1) or right(1.0): " << leftOrRight << std::endl;
                    std::cout << "\t\tdirection: " << direction << std::endl;
                    std::cout << "\t\tarea: " << std::setprecision(16) << area << std::endl;
                    std::cout << "\t\tcomputedNormalArea: [" << normal[0] << ", " << normal[1] << ", " << normal[2] << "]" << std::endl;

                    // Get the stored area
                    PetscFVFaceGeom* fg;
                    DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;

                    std::cout << "\t\tstoredNormalArea: [" << fg->normal[0] << ", " << fg->normal[1] << ", " << fg->normal[2] << "]" << std::endl;

                    std::cout << "\t\tnodes:" << std::endl;
                    // March over each node
                    PetscInt* points = nullptr;
                    PetscInt numPoints;
                    DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &numPoints, &points) >> checkError;

                    for (PetscInt p = 0; p < numPoints; p++) {
                        PetscInt point = points[p * 2];

                        // Check the depth
                        PetscInt pointDepth = 0;
                        DMPlexGetPointDepth(dm, point, &pointDepth) >> checkError;

                        // check if node
                        if (pointDepth == 0) {
                            PetscScalar nodeLocation[3];
                            DMPlexComputeCellGeometryFVM(dm, point, nullptr, nodeLocation, nullptr) >> checkError;
                            std::cout << "\t\t\t" << point << ": [" << nodeLocation[0] << ", " << nodeLocation[1] << ", " << nodeLocation[2] << "]" << std::endl;
                            nodesInCell.insert(point);
                        }
                    }
                }
            }
        }
    });

    // cleanup
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    RestoreRange(dm, faceRange);
    VecDestroy(&faceGeomVec) >> checkError;
    VecDestroy(&cellGeomVec) >> checkError;
}

std::string ablate::domain::modifiers::FvmCheck::ToString() const { return "ablate::domain::modifiers::FvmCheck: " + (region ? region->ToString() : ""); }

void ablate::domain::modifiers::FvmCheck::GetRange(DM dm, PetscInt depth, ablate::solver::Range& range) const {
    // Start out getting all the points
    IS allPointIS;
    DMGetStratumIS(dm, "dim", depth, &allPointIS) >> checkError;
    if (!allPointIS) {
        DMGetStratumIS(dm, "depth", depth, &allPointIS) >> checkError;
    }

    // If there is a label for this solver, get only the parts of the mesh that here
    if (region) {
        DMLabel label;
        DMGetLabel(dm, region->GetName().c_str(), &label);

        IS labelIS;
        DMLabelGetStratumIS(label, region->GetValue(), &labelIS) >> checkError;
        ISIntersect_Caching_Internal(allPointIS, labelIS, &range.is) >> checkError;
        ISDestroy(&labelIS) >> checkError;
    } else {
        PetscObjectReference((PetscObject)allPointIS) >> checkError;
        range.is = allPointIS;
    }

    // Get the point range
    if (range.is == nullptr) {
        // There are no points in this region, so skip
        range.start = 0;
        range.end = 0;
        range.points = nullptr;
    } else {
        // Get the range
        ISGetPointRange(range.is, &range.start, &range.end, &range.points) >> checkError;
    }

    // Clean up the allCellIS
    ISDestroy(&allPointIS) >> checkError;
}
void ablate::domain::modifiers::FvmCheck::RestoreRange(DM, ablate::solver::Range& range) {
    if (range.is) {
        ISRestorePointRange(range.is, &range.start, &range.end, &range.points) >> checkError;
        ISDestroy(&range.is) >> checkError;
    }
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::FvmCheck,
         "The FVM check marches over each face in a mesh region, sums the contributions for each cell in the region to ensure they sum to 0.0",
         OPT(ablate::domain::Region, "region", "the region describing the boundary cells, default is everywhere"),
         OPT(int, "expectedFaceCount", "if specified, the fvmCheck each cell for the correct number of faces"),
         OPT(int, "expectedNodeCount", "if specified, the fvmCheck each cell for the correct number of nodes"));
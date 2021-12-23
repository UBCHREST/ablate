#include "tagLabelInterface.hpp"
#include <utilities/petscError.hpp>
#include <utility>

ablate::domain::modifiers::TagLabelInterface::TagLabelInterface(std::shared_ptr<domain::Region> leftRegion, std::shared_ptr<domain::Region> rightRegion,
                                                                std::shared_ptr<domain::Region> boundaryFaceRegion, std::shared_ptr<domain::Region> leftBoundaryCellRegion,
                                                                std::shared_ptr<domain::Region> rightBoundaryCellRegion)
    : leftRegion(std::move(leftRegion)),
      rightRegion(std::move(rightRegion)),
      boundaryFaceRegion(std::move(boundaryFaceRegion)),
      leftBoundaryCellRegion(std::move(leftBoundaryCellRegion)),
      rightBoundaryCellRegion(std::move(rightBoundaryCellRegion)) {}

void ablate::domain::modifiers::TagLabelInterface::Modify(DM &dm) {
    // get the left label from the dm
    DMLabel leftLabel;
    DMGetLabel(dm, leftRegion->GetName().c_str(), &leftLabel) >> checkError;
    if (leftLabel == nullptr) {
        throw std::runtime_error("Left Label " + leftRegion->GetName() + " cannot be found in the dm.");
    }

    // get the right label from the dm
    DMLabel rightLabel;
    DMGetLabel(dm, rightRegion->GetName().c_str(), &rightLabel) >> checkError;
    if (leftLabel == nullptr) {
        throw std::runtime_error("Right Label " + rightRegion->GetName() + " cannot be found in the dm.");
    }

    // Create a new label
    DMCreateLabel(dm, boundaryFaceRegion->GetName().c_str()) >> checkError;
    DMLabel boundaryFaceLabel;
    DMGetLabel(dm, boundaryFaceRegion->GetName().c_str(), &boundaryFaceLabel) >> checkError;

    // create cell labels if requested
    DMLabel leftBoundaryCellLabel = nullptr;
    PetscInt leftBoundaryCellValue = 0;
    if (leftBoundaryCellRegion) {
        DMCreateLabel(dm, leftBoundaryCellRegion->GetName().c_str()) >> checkError;
        DMGetLabel(dm, leftBoundaryCellRegion->GetName().c_str(), &leftBoundaryCellLabel) >> checkError;
        leftBoundaryCellValue = leftBoundaryCellRegion->GetValue();
    }
    DMLabel rightBoundaryCellLabel = nullptr;
    PetscInt rightBoundaryCellValue = 0;
    if (rightBoundaryCellRegion) {
        DMCreateLabel(dm, rightBoundaryCellRegion->GetName().c_str()) >> checkError;
        DMGetLabel(dm, rightBoundaryCellRegion->GetName().c_str(), &rightBoundaryCellLabel) >> checkError;
        rightBoundaryCellValue = rightBoundaryCellRegion->GetValue();
    }

    // Get all the faces
    PetscInt depth;
    DMPlexGetDepth(dm, &depth) >> checkError;
    depth -= 1;
    IS allFacesIS;
    DMGetStratumIS(dm, "dim", depth, &allFacesIS) >> checkError;
    if (!allFacesIS) {
        DMGetStratumIS(dm, "depth", depth, &allFacesIS) >> checkError;
    }

    // Get all the points in the label
    IS leftLabelIS;
    DMLabelGetStratumIS(leftLabel, leftRegion->GetValue(), &leftLabelIS) >> checkError;

    // Get the intersect between these two IS
    IS leftFaceIS;
    ISIntersect(allFacesIS, leftLabelIS, &leftFaceIS) >> checkError;

    // Get the face range
    PetscInt fStart, fEnd;
    const PetscInt *faces;
    ISGetPointRange(leftFaceIS, &fStart, &fEnd, &faces) >> checkError;

    // March over each face
    for (PetscInt f = fStart; f < fEnd; ++f) {
        const PetscInt face = faces ? faces[f] : f;

        // Determine the number of supports (for faces, the support would be the cell/element(
        PetscInt supportSize;
        DMPlexGetSupportSize(dm, face, &supportSize);

        if (supportSize == 2) {
            // Check if any of the supports/cells are not in the label.  This will tell us if this is a boundary face
            const PetscInt *cells;
            DMPlexGetSupport(dm, face, &cells) >> checkError;

            // Check to see if 0 cell is in left
            PetscInt zeroCellLabelLeftValue, oneCellLeftValue;
            DMLabelGetValue(leftLabel, cells[0], &zeroCellLabelLeftValue) >> checkError;
            DMLabelGetValue(leftLabel, cells[1], &oneCellLeftValue) >> checkError;

            if (zeroCellLabelLeftValue == leftRegion->GetValue()) {
                // Check if the 1 cell is in the right region
                PetscInt otherCellValue;
                DMLabelGetValue(rightLabel, cells[1], &otherCellValue) >> checkError;
                if (otherCellValue == rightRegion->GetValue()) {
                    DMLabelSetValue(boundaryFaceLabel, face, boundaryFaceRegion->GetValue()) >> checkError;

                    if (leftBoundaryCellLabel) {
                        DMLabelSetValue(leftBoundaryCellLabel, cells[0], leftBoundaryCellValue) >> checkError;
                    }
                    if (rightBoundaryCellLabel) {
                        DMLabelSetValue(rightBoundaryCellLabel, cells[1], rightBoundaryCellValue) >> checkError;
                    }
                }
            } else if (oneCellLeftValue == leftRegion->GetValue()) {
                // check if the 1 cell is in the left region
                PetscInt otherCellValue;
                DMLabelGetValue(rightLabel, cells[0], &otherCellValue) >> checkError;
                if (otherCellValue == rightRegion->GetValue()) {
                    DMLabelSetValue(boundaryFaceLabel, face, boundaryFaceRegion->GetValue()) >> checkError;

                    if (leftBoundaryCellLabel) {
                        DMLabelSetValue(leftBoundaryCellLabel, cells[1], leftBoundaryCellValue) >> checkError;
                    }
                    if (rightBoundaryCellLabel) {
                        DMLabelSetValue(rightBoundaryCellLabel, cells[0], rightBoundaryCellValue) >> checkError;
                    }
                }
            }
        }
    }
    DistributeLabel(dm, boundaryFaceLabel);
    if(rightBoundaryCellLabel) {
        DistributeLabel(dm, rightBoundaryCellLabel);
    }
    if(leftBoundaryCellLabel) {
        DistributeLabel(dm, leftBoundaryCellLabel);
    }
    ISRestorePointRange(leftFaceIS, &fStart, &fEnd, &faces) >> checkError;
    ISDestroy(&leftFaceIS);
    ISDestroy(&leftLabelIS);
    ISDestroy(&allFacesIS);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::TagLabelInterface,
         "Class to label/tag all faces/cells on the interface between two labels.  The left/right designations are just used to separate the left/right labels.",
         ARG(ablate::domain::Region, "leftRegion", "the \"left\" region"), ARG(ablate::domain::Region, "rightRegion", "the \"right\" region"),
         ARG(ablate::domain::Region, "boundaryFaceRegion", "the new region for the newly tagged boundary faces"),
         OPT(ablate::domain::Region, "leftBoundaryCellRegion", "optional new region to tag the boundary cells on the \"left\" of region"),
         OPT(ablate::domain::Region, "rightBoundaryCellRegion", "optional new region to tag the boundary cells on the \"right\" of region"));
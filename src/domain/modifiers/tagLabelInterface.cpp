#include "tagLabelInterface.hpp"
#include "utilities/petscUtilities.hpp"
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
    DMGetLabel(dm, leftRegion->GetName().c_str(), &leftLabel) >> utilities::PetscUtilities::checkError;
    if (leftLabel == nullptr) {
        throw std::runtime_error("Left Label " + leftRegion->GetName() + " cannot be found in the dm.");
    }

    // get the right label from the dm
    DMLabel rightLabel;
    DMGetLabel(dm, rightRegion->GetName().c_str(), &rightLabel) >> utilities::PetscUtilities::checkError;
    if (leftLabel == nullptr) {
        throw std::runtime_error("Right Label " + rightRegion->GetName() + " cannot be found in the dm.");
    }

    // Create a new label
    DMCreateLabel(dm, boundaryFaceRegion->GetName().c_str()) >> utilities::PetscUtilities::checkError;
    DMLabel boundaryFaceLabel;
    DMGetLabel(dm, boundaryFaceRegion->GetName().c_str(), &boundaryFaceLabel) >> utilities::PetscUtilities::checkError;

    // create cell labels if requested
    DMLabel leftBoundaryCellLabel = nullptr;
    PetscInt leftBoundaryCellValue = 0;
    if (leftBoundaryCellRegion) {
        DMCreateLabel(dm, leftBoundaryCellRegion->GetName().c_str()) >> utilities::PetscUtilities::checkError;
        DMGetLabel(dm, leftBoundaryCellRegion->GetName().c_str(), &leftBoundaryCellLabel) >> utilities::PetscUtilities::checkError;
        leftBoundaryCellValue = leftBoundaryCellRegion->GetValue();
    }
    DMLabel rightBoundaryCellLabel = nullptr;
    PetscInt rightBoundaryCellValue = 0;
    if (rightBoundaryCellRegion) {
        DMCreateLabel(dm, rightBoundaryCellRegion->GetName().c_str()) >> utilities::PetscUtilities::checkError;
        DMGetLabel(dm, rightBoundaryCellRegion->GetName().c_str(), &rightBoundaryCellLabel) >> utilities::PetscUtilities::checkError;
        rightBoundaryCellValue = rightBoundaryCellRegion->GetValue();
    }

    // Get all the faces
    PetscInt depth;
    DMPlexGetDepth(dm, &depth) >> utilities::PetscUtilities::checkError;
    depth -= 1;
    IS allFacesIS;
    DMGetStratumIS(dm, "dim", depth, &allFacesIS) >> utilities::PetscUtilities::checkError;
    if (!allFacesIS) {
        DMGetStratumIS(dm, "depth", depth, &allFacesIS) >> utilities::PetscUtilities::checkError;
    }

    // Get all the points in the label
    IS leftLabelIS;
    DMLabelGetStratumIS(leftLabel, leftRegion->GetValue(), &leftLabelIS) >> utilities::PetscUtilities::checkError;

    // Get the intersect between these two IS
    if (leftLabelIS) {
        IS leftFaceIS;
        ISIntersect(allFacesIS, leftLabelIS, &leftFaceIS) >> utilities::PetscUtilities::checkError;

        // Get the face range
        PetscInt fStart, fEnd;
        const PetscInt *faces;
        ISGetPointRange(leftFaceIS, &fStart, &fEnd, &faces) >> utilities::PetscUtilities::checkError;

        // March over each face
        for (PetscInt f = fStart; f < fEnd; ++f) {
            const PetscInt face = faces ? faces[f] : f;

            // Determine the number of supports (for faces, the support would be the cell/element(
            PetscInt supportSize;
            DMPlexGetSupportSize(dm, face, &supportSize);

            if (supportSize == 2) {
                // Check if any of the supports/cells are not in the label.  This will tell us if this is a boundary face
                const PetscInt *cells;
                DMPlexGetSupport(dm, face, &cells) >> utilities::PetscUtilities::checkError;

                // Check to see if 0 cell is in left
                PetscInt zeroCellLabelLeftValue, oneCellLeftValue;
                DMLabelGetValue(leftLabel, cells[0], &zeroCellLabelLeftValue) >> utilities::PetscUtilities::checkError;
                DMLabelGetValue(leftLabel, cells[1], &oneCellLeftValue) >> utilities::PetscUtilities::checkError;

                if (zeroCellLabelLeftValue == leftRegion->GetValue()) {
                    // Check if the 1 cell is in the right region
                    PetscInt otherCellValue;
                    DMLabelGetValue(rightLabel, cells[1], &otherCellValue) >> utilities::PetscUtilities::checkError;
                    if (otherCellValue == rightRegion->GetValue()) {
                        DMLabelSetValue(boundaryFaceLabel, face, boundaryFaceRegion->GetValue()) >> utilities::PetscUtilities::checkError;

                        if (leftBoundaryCellLabel) {
                            DMLabelSetValue(leftBoundaryCellLabel, cells[0], leftBoundaryCellValue) >> utilities::PetscUtilities::checkError;
                        }
                        if (rightBoundaryCellLabel) {
                            DMLabelSetValue(rightBoundaryCellLabel, cells[1], rightBoundaryCellValue) >> utilities::PetscUtilities::checkError;
                        }
                    }
                } else if (oneCellLeftValue == leftRegion->GetValue()) {
                    // check if the 1 cell is in the left region
                    PetscInt otherCellValue;
                    DMLabelGetValue(rightLabel, cells[0], &otherCellValue) >> utilities::PetscUtilities::checkError;
                    if (otherCellValue == rightRegion->GetValue()) {
                        DMLabelSetValue(boundaryFaceLabel, face, boundaryFaceRegion->GetValue()) >> utilities::PetscUtilities::checkError;

                        if (leftBoundaryCellLabel) {
                            DMLabelSetValue(leftBoundaryCellLabel, cells[1], leftBoundaryCellValue) >> utilities::PetscUtilities::checkError;
                        }
                        if (rightBoundaryCellLabel) {
                            DMLabelSetValue(rightBoundaryCellLabel, cells[0], rightBoundaryCellValue) >> utilities::PetscUtilities::checkError;
                        }
                    }
                }
            }
        }
        ISRestorePointRange(leftFaceIS, &fStart, &fEnd, &faces) >> utilities::PetscUtilities::checkError;
        ISDestroy(&leftFaceIS);
        ISDestroy(&leftLabelIS);
    }

    // check to see if we should distribute the labels
    PetscBool isDistributed;
    DMPlexIsDistributed(dm, &isDistributed) >> utilities::PetscUtilities::checkError;
    if (isDistributed) {
        DistributeLabel(dm, boundaryFaceLabel);
        if (rightBoundaryCellLabel) {
            DistributeLabel(dm, rightBoundaryCellLabel);
        }
        if (leftBoundaryCellLabel) {
            DistributeLabel(dm, leftBoundaryCellLabel);
        }
    }
    ISDestroy(&allFacesIS);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::TagLabelInterface,
         "Class to label/tag all faces/cells on the interface between two labels.  The left/right designations are just used to separate the left/right labels.",
         ARG(ablate::domain::Region, "leftRegion", "the \"left\" region"), ARG(ablate::domain::Region, "rightRegion", "the \"right\" region"),
         ARG(ablate::domain::Region, "boundaryFaceRegion", "the new region for the newly tagged boundary faces"),
         OPT(ablate::domain::Region, "leftBoundaryCellRegion", "optional new region to tag the boundary cells on the \"left\" of region"),
         OPT(ablate::domain::Region, "rightBoundaryCellRegion", "optional new region to tag the boundary cells on the \"right\" of region"));
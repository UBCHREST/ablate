#include "tagLabelBoundary.hpp"
#include <utilities/petscError.hpp>

ablate::domain::modifiers::TagLabelBoundary::TagLabelBoundary(std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> boundaryFaceRegion,
                                                              const std::shared_ptr<domain::Region> boundaryCellRegion)
    : region(region), boundaryFaceRegion(boundaryFaceRegion), boundaryCellRegion(boundaryCellRegion) {}

void ablate::domain::modifiers::TagLabelBoundary::Modify(DM &dm) {
    DMLabel label;
    DMGetLabel(dm, region->GetName().c_str(), &label) >> checkError;
    if (label == nullptr) {
        throw std::runtime_error("Label " + region->GetName() + " cannot be found in the dm.");
    }

    // Create a new label
    DMCreateLabel(dm, boundaryFaceRegion->GetName().c_str()) >> checkError;
    DMLabel boundaryFaceLabel;
    DMGetLabel(dm, boundaryFaceRegion->GetName().c_str(), &boundaryFaceLabel) >> checkError;

    DMLabel boundaryCellLabel = nullptr;
    if (boundaryCellRegion) {
        DMCreateLabel(dm, boundaryCellRegion->GetName().c_str()) >> checkError;
        DMGetLabel(dm, boundaryCellRegion->GetName().c_str(), &boundaryCellLabel) >> checkError;
    }

    // Get all of the faces
    PetscInt depth;
    DMPlexGetDepth(dm, &depth) >> checkError;
    depth -= 1;
    IS allFacesIS;
    DMGetStratumIS(dm, "dim", depth, &allFacesIS) >> checkError;
    if (!allFacesIS) {
        DMGetStratumIS(dm, "depth", depth, &allFacesIS) >> checkError;
    }

    // Get all of the points in the label
    IS labelIS;
    DMLabelGetStratumIS(label, region->GetValue(), &labelIS) >> checkError;

    // Get the intersect between these two IS
    IS faceIS;
    ISIntersect(allFacesIS, labelIS, &faceIS) >> checkError;

    // Get the face range
    PetscInt fStart, fEnd;
    const PetscInt *faces;
    ISGetPointRange(faceIS, &fStart, &fEnd, &faces) >> checkError;

    // March over each face
    for (PetscInt f = fStart; f < fEnd; ++f) {
        const PetscInt face = faces ? faces[f] : f;

        // Determine the number of supports (for faces, the support would be the cell/element(
        PetscInt supportSize;
        DMPlexGetSupportSize(dm, face, &supportSize);

        if (supportSize == 1) {
            // Assume that this face is a boundary
            DMLabelSetValue(label, face, boundaryFaceRegion->GetValue()) >> checkError;
        } else {
            // Check if any of the supports/cells are not in the label.  This will tell us if this is a boundary face
            const PetscInt *cells;
            DMPlexGetSupport(dm, face, &cells) >> checkError;

            for (PetscInt c = 0; c < supportSize; ++c) {
                const PetscInt cell = cells[c];
                PetscInt cellLabelValue;
                DMLabelGetValue(label, cell, &cellLabelValue) >> checkError;
                // If this cell is not inside of original region
                if (cellLabelValue != region->GetValue()) {
                    // tage the face cell
                    DMLabelSetValue(boundaryFaceLabel, face, boundaryFaceRegion->GetValue()) >> checkError;

                    // also tag the boundary cell
                    if (boundaryCellLabel) {
                        DMLabelSetValue(boundaryCellLabel, cell, boundaryCellRegion->GetValue()) >> checkError;
                    }
                }
            }
        }
    }

    DistributeLabel(dm, boundaryFaceLabel);
    if(boundaryCellLabel) {
        DistributeLabel(dm, boundaryCellLabel);
    }

    ISRestorePointRange(faceIS, &fStart, &fEnd, &faces) >> checkError;
    ISDestroy(&faceIS) >> checkError;
    ISDestroy(&labelIS) >> checkError;
    ISDestroy(&allFacesIS) >> checkError;
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::TagLabelBoundary, "Creates a new label for all faces on the outside of the boundary",
         ARG(ablate::domain::Region, "region", "the region to tag the boundary"), ARG(ablate::domain::Region, "boundaryFaceRegion", "the new region for the boundary faces"),
         OPT(ablate::domain::Region, "boundaryCellRegion", "the new region for the boundary cells"));
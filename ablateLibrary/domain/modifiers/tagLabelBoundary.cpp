#include "tagLabelBoundary.hpp"
#include <utilities/petscError.hpp>

ablate::domain::modifier::TagLabelBoundary::TagLabelBoundary(std::string name, std::string boundaryName, int labelValueIn, int boundaryLabelValueIn)
    : name(name), boundaryName(boundaryName), labelValue(labelValueIn == 0 ? 1 : (PetscInt)labelValueIn), boundaryLabelValue(boundaryLabelValueIn == 0 ? 1 : (PetscInt)boundaryLabelValueIn) {}

void ablate::domain::modifier::TagLabelBoundary::Modify(DM &dm) {
    DMLabel label;
    DMGetLabel(dm, name.c_str(), &label) >> checkError;
    if (label == nullptr) {
        throw std::runtime_error("Label " + name + " cannot be found in the dm.");
    }

    // Create a new label
    DMCreateLabel(dm, boundaryName.c_str()) >> checkError;
    DMLabel boundaryLabel;
    DMGetLabel(dm, boundaryName.c_str(), &boundaryLabel) >> checkError;

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
    DMLabelGetStratumIS(label, labelValue, &labelIS) >> checkError;

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
            DMLabelSetValue(label, face, boundaryLabelValue) >> checkError;
        } else {
            // Check if any of the supports/cells are not in the label.  This will tell us if this is a boundary face
            const PetscInt *cells;
            DMPlexGetSupport(dm, face, &cells) >> checkError;

            for (PetscInt c = 0; c < supportSize; ++c) {
                const PetscInt cell = cells[c];
                PetscInt cellLabelValue;
                DMLabelGetValue(label, cell, &cellLabelValue) >> checkError;
                if (cellLabelValue != labelValue) {
                    DMLabelSetValue(boundaryLabel, face, boundaryLabelValue) >> checkError;
                }
            }
        }
    }

    ISRestorePointRange(faceIS, &fStart, &fEnd, &faces) >> checkError;
    ISDestroy(&faceIS);
    ISDestroy(&labelIS);
    ISDestroy(&allFacesIS);
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::modifier::Modifier, ablate::domain::modifier::TagLabelBoundary, "Creates a new label for all faces on the ouside of the boundary",
         ARG(std::string, "name", "the field label name"), ARG(std::string, "boundaryName", "the new boundary label name"), OPT(int, "labelValue", "The label value, default is 1"),
         OPT(int, "boundaryLabelValue", "The label value for the new boundary"));
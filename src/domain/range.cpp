#include "range.hpp"
#include "utilities/petscUtilities.hpp"
#include <petsc/private/dmpleximpl.h> // For ISIntersect_Caching_Internal, used in ablate::domain::SubDomain::GetRange

void ablate::domain::GetRange(DM dm, const std::shared_ptr<ablate::domain::Region> region, PetscInt depth, ablate::domain::Range &range) {
    // Start out getting all the points
    IS allPointIS;
    DMGetStratumIS(dm, "dim", depth, &allPointIS) >> utilities::PetscUtilities::checkError;
    if (!allPointIS) {
        DMGetStratumIS(dm, "depth", depth, &allPointIS) >> utilities::PetscUtilities::checkError;
    }

    // If there is a label for this solver, get only the parts of the mesh that here
    if (region) {
        DMLabel label;
        DMGetLabel(dm, region->GetName().c_str(), &label);

        IS labelIS;
        DMLabelGetStratumIS(label, region->GetValue(), &labelIS) >> utilities::PetscUtilities::checkError;
        ISIntersect_Caching_Internal(allPointIS, labelIS, &range.is) >> utilities::PetscUtilities::checkError;
        ISDestroy(&labelIS) >> utilities::PetscUtilities::checkError;
    } else {
        PetscObjectReference((PetscObject)allPointIS) >> utilities::PetscUtilities::checkError;
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
        ISGetPointRange(range.is, &range.start, &range.end, &range.points) >> utilities::PetscUtilities::checkError;
    }

    // Clean up the allCellIS
    ISDestroy(&allPointIS) >> utilities::PetscUtilities::checkError;
}

void ablate::domain::GetCellRange(DM dm, const std::shared_ptr<ablate::domain::Region> region, ablate::domain::Range &cellRange) {
    // Start out getting all the cells
    PetscInt depth;
    DMPlexGetDepth(dm, &depth) >> utilities::PetscUtilities::checkError;
    ablate::domain::GetRange(dm, region, depth, cellRange);
}


void ablate::domain::GetFaceRange(DM dm, const std::shared_ptr<ablate::domain::Region> region, ablate::domain::Range &faceRange) {
    // Start out getting all the faces
    PetscInt depth;
    DMPlexGetDepth(dm, &depth) >> utilities::PetscUtilities::checkError;
    ablate::domain::GetRange(dm, region, depth - 1, faceRange);
}

void ablate::domain::RestoreRange(ablate::domain::Range &range) {
    if (range.is) {
        ISRestorePointRange(range.is, &range.start, &range.end, &range.points) >> utilities::PetscUtilities::checkError;
        ISDestroy(&range.is) >> utilities::PetscUtilities::checkError;
    }
}


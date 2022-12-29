#include "intersectLabels.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::modifiers::IntersectLabels::IntersectLabels(std::shared_ptr<domain::Region> intersectRegion, std::vector<std::shared_ptr<domain::Region>> regions)
    : intersectRegion(intersectRegion), regions(regions) {}

void ablate::domain::modifiers::IntersectLabels::Modify(DM& dm) {
    // Create a new label for the merged region
    DMCreateLabel(dm, intersectRegion->GetName().c_str()) >> utilities::PetscUtilities::checkError;

    // Get the is and intersect
    IS intersectIS;
    DMGetStratumIS(dm, regions[0]->GetName().c_str(), regions[0]->GetValue(), &intersectIS) >> utilities::PetscUtilities::checkError;

    for (std::size_t r = 1; r < regions.size(); r++) {
        IS otherIS = nullptr;
        DMGetStratumIS(dm, regions[r]->GetName().c_str(), regions[r]->GetValue(), &otherIS) >> utilities::PetscUtilities::checkError;

        // Get the intersection
        IS newMergedIS;
        ISIntersect(intersectIS, otherIS, &newMergedIS) >> utilities::PetscUtilities::checkError;

        // destroy the other and old is
        ISDestroy(&intersectIS) >> utilities::PetscUtilities::checkError;
        ISDestroy(&otherIS) >> utilities::PetscUtilities::checkError;
        intersectIS = newMergedIS;
    }

    DMLabel intersectLabel;
    DMGetLabel(dm, intersectRegion->GetName().c_str(), &intersectLabel) >> utilities::PetscUtilities::checkError;
    DMLabelSetStratumIS(intersectLabel, intersectRegion->GetValue(), intersectIS) >> utilities::PetscUtilities::checkError;

    // check the size
    PetscInt newLabelSize;
    ISGetSize(intersectIS, &newLabelSize) >> utilities::PetscUtilities::checkError;
    if (newLabelSize == 0) {
        throw std::length_error("The new intersect region " + intersectRegion->GetName() + " resulted in no points.");
    }

    // cleanup
    ISDestroy(&intersectIS) >> utilities::PetscUtilities::checkError;
    DMPlexLabelComplete(dm, intersectLabel) >> utilities::PetscUtilities::checkError;
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::IntersectLabels, "Creates a new label that intersections the provided regions",
         ARG(ablate::domain::Region, "intersectRegion", "the intersect region to create/used"), ARG(std::vector<ablate::domain::Region>, "regions", "the regions to include in the intersection"));

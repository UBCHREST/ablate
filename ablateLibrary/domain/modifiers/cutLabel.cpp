#include "cutLabel.hpp"
#include "utilities/petscError.hpp"

ablate::domain::modifiers::CutLabel::CutLabel(std::shared_ptr<domain::Region> cutRegion, std::vector<std::shared_ptr<domain::Region>> regions) : cutRegion(cutRegion), regions(regions) {}

void ablate::domain::modifiers::CutLabel::Modify(DM& dm) {
    // Get the label for each region
    std::vector<IS> regionISs(regions.size(), nullptr);

    // Get the data
    for (std::size_t r = 0; r < regions.size(); r++) {
        auto& regionIS = regionISs[r];
        DMGetStratumIS(dm, regions[r]->GetName().c_str(), regions[r]->GetValue(), &regionIS) >> checkError;
    }
    // Create Concatenate IS that is all the cut regions
    IS mergedIS;
    ISConcatenate(PetscObjectComm((PetscObject)dm), regionISs.size(), regionISs.data(), &mergedIS) >> checkError;
    ISSortRemoveDups(mergedIS) >> checkError;

    // cleanup
    for (auto& is : regionISs) {
        ISDestroy(&is) >> checkError;
    }

    // get the cut region
    DMLabel cutRegionLabel;
    IS orgIS;
    DMGetLabel(dm, cutRegion->GetName().c_str(), &cutRegionLabel) >> checkError;
    DMLabelGetStratumIS(cutRegionLabel, cutRegion->GetValue(), &orgIS) >> checkError;

    // compute the new cut region
    IS cutIS;
    ISDifference(orgIS, mergedIS, &cutIS) >> checkError;
    DMLabelSetStratumIS(cutRegionLabel, cutRegion->GetValue(), cutIS) >> checkError;

    PetscSF         sfPoint;
    PetscInt nroots;
    DMGetPointSF(dm, &sfPoint);
    PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL) >> checkError;
    printf("nroots: %d\n", (int)nroots) >> checkError;

    // cleanup
    ISDestroy(&cutIS) >> checkError;
    ISDestroy(&orgIS) >> checkError;
    ISDestroy(&mergedIS) >> checkError;
    DMPlexLabelComplete(dm, cutRegionLabel) >> checkError;
}
std::string ablate::domain::modifiers::CutLabel::ToString() const {
    std::string string = "ablate::domain::modifiers::CutLabel\n";
    string += "\tcutRegion: " + cutRegion->ToString() + "\n";
    string += "\tregions:\n";
    for (const auto& region : regions) {
        string += "\t\t" + region->GetName() + "\n";
    }
    string.pop_back();
    return string;
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::CutLabel, "Cuts/removes the given regions out of the cutRegion",
         ARG(ablate::domain::Region, "cutRegion", "the merged region to cut"), ARG(std::vector<ablate::domain::Region>, "regions", "the regions to be removed in the cut region"));

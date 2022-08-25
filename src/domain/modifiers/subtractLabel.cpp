#include "subtractLabel.hpp"
#include "utilities/petscError.hpp"

ablate::domain::modifiers::SubtractLabel::SubtractLabel(std::shared_ptr<domain::Region> differenceRegion, std::shared_ptr<domain::Region> minuendRegion,
                                                        std::vector<std::shared_ptr<domain::Region>> subtrahendRegions, bool incompleteLabel)
    : differenceRegion(differenceRegion), minuendRegion(minuendRegion), subtrahendRegions(subtrahendRegions), incompleteLabel(incompleteLabel) {}

void ablate::domain::modifiers::SubtractLabel::Modify(DM& dm) {
    // Create a new label
    DMCreateLabel(dm, differenceRegion->GetName().c_str()) >> checkError;
    DMLabel differenceLabel;
    DMGetLabel(dm, differenceRegion->GetName().c_str(), &differenceLabel) >> checkError;

    // get the minuendIS
    minuendRegion->CheckForLabel(dm);
    DMLabel minuendLabel = nullptr;
    DMGetLabel(dm, minuendRegion->GetName().c_str(), &minuendLabel) >> checkError;
    IS minuendIS = nullptr;
    DMLabelGetStratumIS(minuendLabel, minuendRegion->GetValue(), &minuendIS) >> checkError;

    // build the list of is to remove
    std::vector<IS> isList;
    for (const auto& subtrahendRegion : subtrahendRegions) {
        subtrahendRegion->CheckForLabel(dm);
        DMLabel subtrahendLabel;
        DMGetLabel(dm, subtrahendRegion->GetName().c_str(), &subtrahendLabel) >> checkError;
        IS subtrahendIS = nullptr;
        DMLabelGetStratumIS(subtrahendLabel, subtrahendRegion->GetValue(), &subtrahendIS) >> checkError;
        isList.push_back(subtrahendIS);
    }

    // Merge the regions
    IS subtrahendsIS = nullptr;
    ISConcatenate(PetscObjectComm((PetscObject)minuendIS), isList.size(), isList.data(), &subtrahendsIS) >> checkError;
    ISSortRemoveDups(subtrahendsIS) >> checkError;

    // compute the difference
    IS differenceIS = nullptr;

    if (minuendIS) {
        ISDifference(minuendIS, subtrahendsIS, &differenceIS) >> checkError;
    }

    // If the differenceIS is defined, apply it to the region
    if (differenceIS) {
        DMLabelSetStratumIS(differenceLabel, differenceRegion->GetValue(), differenceIS) >> checkError;
    }

    if (!incompleteLabel) {
        DMPlexLabelComplete(dm, differenceLabel) >> checkError;
    }

    // cleanup
    if (differenceIS) {
        ISDestroy(&differenceIS) >> checkError;
    }
    if (minuendIS) {
        ISDestroy(&minuendIS) >> checkError;
    }
    for (auto& subtrahendIS : isList) {
        ISDestroy(&subtrahendIS) >> checkError;
    }
    if (subtrahendsIS) {
        ISDestroy(&subtrahendsIS) >> checkError;
    }
}
std::string ablate::domain::modifiers::SubtractLabel::ToString() const {
    std::string string = "ablate::domain::modifiers::SubtractLabel\n";
    string += "\tdifferenceRegion: " + differenceRegion->ToString() + "\n";
    string += "\tminuendRegion: " + minuendRegion->ToString() + "\n";
    string += "\tsubtrahendRegion(s): \n";
    for (const auto& region : subtrahendRegions) {
        string += "\t\t " + region->ToString() + " \n";
    }
    return string;
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::SubtractLabel, "Cuts/removes the given region (difference = minuend - subtrahend)",
         ARG(ablate::domain::Region, "differenceRegion", "the result of the operation"), ARG(ablate::domain::Region, "minuendRegion", "the minuend region"),
         ARG(std::vector<ablate::domain::Region>, "subtrahendRegions", "the region(s) to be removed"),
         OPT(bool, "incompleteLabel",
             "determines if the DMPlexLabelComplete function for the new label is called. (true = DMPlexLabelComplete not called, false = DMPlexLabelComplete called, default is false"));

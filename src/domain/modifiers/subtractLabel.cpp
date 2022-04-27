#include "subtractLabel.hpp"
#include "utilities/petscError.hpp"

ablate::domain::modifiers::SubtractLabel::SubtractLabel(std::shared_ptr<domain::Region> differenceRegion, std::shared_ptr<domain::Region> minuendRegion,
                                                        std::shared_ptr<domain::Region> subtrahendRegion)
    : differenceRegion(differenceRegion), minuendRegion(minuendRegion), subtrahendRegion(subtrahendRegion) {}

void ablate::domain::modifiers::SubtractLabel::Modify(DM& dm) {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
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

    // Get the subtrahendIS
    DMLabel subtrahendLabel = nullptr;
    subtrahendRegion->CheckForLabel(dm);
    DMGetLabel(dm, subtrahendRegion->GetName().c_str(), &subtrahendLabel) >> checkError;
    IS subtrahendIS = nullptr;
    DMLabelGetStratumIS(subtrahendLabel, subtrahendRegion->GetValue(), &subtrahendIS) >> checkError;

    // compute the difference
    IS differenceIS = nullptr;

    if (minuendIS) {
        ISDifference(minuendIS, subtrahendIS, &differenceIS) >> checkError;
    }

    // If the differenceIS is defined, apply it to the region
    if (differenceIS) {
        DMLabelSetStratumIS(differenceLabel, differenceRegion->GetValue(), differenceIS) >> checkError;
    }

    // cleanup
    if (differenceIS) {
        ISDestroy(&differenceIS) >> checkError;
    }
    if (minuendIS) {
        ISDestroy(&minuendIS) >> checkError;
    }
    if (subtrahendIS) {
        ISDestroy(&subtrahendIS) >> checkError;
    }
}
std::string ablate::domain::modifiers::SubtractLabel::ToString() const {
    std::string string = "ablate::domain::modifiers::SubtractLabel\n";
    string += "\tdifferenceRegion: " + differenceRegion->ToString() + "\n";
    string += "\tminuendRegion: " + minuendRegion->ToString() + "\n";
    string += "\tsubtrahendRegion: " + subtrahendRegion->ToString() + "\n";
    return string;
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::SubtractLabel, "Cuts/removes the given region (difference = minuend - subtrahend)",
         ARG(ablate::domain::Region, "differenceRegion", "the result of the operation"), ARG(ablate::domain::Region, "minuendRegion", "the minuend region"),
         ARG(ablate::domain::Region, "subtrahendRegion", "the region to be removed"));

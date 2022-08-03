#include "completeLabel.hpp"
#include <utilities/petscError.hpp>

ablate::domain::modifiers::CompleteLabel::CompleteLabel(std::shared_ptr<domain::Region> region) : region(region) {}

void ablate::domain::modifiers::CompleteLabel::Modify(DM &dm) {
    DMLabel completeLabel = nullptr;
    DMGetLabel(dm, region->GetName().c_str(), &completeLabel) >> checkError;

    if (completeLabel) {
        DMPlexLabelComplete(dm, completeLabel) >> checkError;
    }
}
std::string ablate::domain::modifiers::CompleteLabel::ToString() const { return "ablate::domain::modifiers::CreateLabel: " + region->ToString(); }

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::CompleteLabel,
         "Wrapper for [DMPlexLabelComplete](https://petsc.org/release/docs/manualpages/DMPLEX/DMPlexLabelComplete.html).  Complete the labels; such that if your label includes all faces, all "
         "vertices connected are also labeled.",
         ARG(ablate::domain::Region, "region", "the region describing the label to complete"));
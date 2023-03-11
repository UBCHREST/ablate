#include "collapseLabels.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::modifiers::CollapseLabels::CollapseLabels(std::vector<std::shared_ptr<domain::Region>> regions) : regions(regions) {}

void ablate::domain::modifiers::CollapseLabels::Modify(DM& dm) {
    // march over each region
    for (const auto& region : regions) {
        // Get the associated label
        DMLabel label;
        PetscInt regionValue;
        domain::Region::GetLabel(region, dm, label, regionValue);

        if (label) {
            // determine the default value for the label
            PetscInt defaultValue;
            DMLabelGetDefaultValue(label, &defaultValue) >> utilities::PetscUtilities::checkError;

            // March over each point in the dm
            PetscInt pStart, pEnd;
            DMPlexGetChart(dm, &pStart, &pEnd) >> utilities::PetscUtilities::checkError;
            for (PetscInt p = pStart; p < pEnd; ++p) {
                PetscInt value;
                DMLabelGetValue(label, p, &value) >> utilities::PetscUtilities::checkError;

                // as long as it is not the default value, set it to the regionValue
                if (value != defaultValue) {
                    DMLabelClearValue(label, p, value) >> utilities::PetscUtilities::checkError;
                    DMLabelSetValue(label, p, regionValue) >> utilities::PetscUtilities::checkError;
                }
            }
            DMPlexLabelComplete(dm, label) >> utilities::PetscUtilities::checkError;
        }
    }
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::CollapseLabels, "Collapse all set values in a label to the provided value in each region. This also completes each label.",
         ARG(std::vector<ablate::domain::Region>, "regions", "the regions to collapse"));

#include "modifier.hpp"
#include <petsc/private/dmpleximpl.h>
#include "utilities/petscUtilities.hpp"

std::ostream& ablate::domain::modifiers::operator<<(std::ostream& os, const ablate::domain::modifiers::Modifier& modifier) {
    os << modifier.ToString();
    return os;
}
void ablate::domain::modifiers::Modifier::ReplaceDm(DM& originalDm, DM& replaceDm) {
    if (replaceDm) {
        // copy over the name
        const char* name;
        PetscObjectName((PetscObject)originalDm) >> utilities::PetscUtilities::checkError;
        PetscObjectGetName((PetscObject)originalDm, &name) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)replaceDm, name) >> utilities::PetscUtilities::checkError;

        // Copy over the options object
        PetscOptions options;
        PetscObjectGetOptions((PetscObject)originalDm, &options) >> utilities::PetscUtilities::checkError;
        PetscObjectSetOptions((PetscObject)replaceDm, options) >> utilities::PetscUtilities::checkError;
        ((DM_Plex*)(replaceDm)->data)->useHashLocation = ((DM_Plex*)originalDm->data)->useHashLocation;

        DMDestroy(&originalDm) >> utilities::PetscUtilities::checkError;
        originalDm = replaceDm;
    }
}

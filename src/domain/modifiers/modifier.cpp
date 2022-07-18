#include "modifier.hpp"
#include <petsc/private/dmpleximpl.h>
#include "utilities/petscError.hpp"

std::ostream& ablate::domain::modifiers::operator<<(std::ostream& os, const ablate::domain::modifiers::Modifier& modifier) {
    os << modifier.ToString();
    return os;
}
void ablate::domain::modifiers::Modifier::ReplaceDm(DM& originalDm, DM& replaceDm) {
    if (replaceDm) {
        // copy over the name
        const char* name;
        PetscObjectName((PetscObject)originalDm) >> ablate::checkError;
        PetscObjectGetName((PetscObject)originalDm, &name) >> ablate::checkError;
        PetscObjectSetName((PetscObject)replaceDm, name) >> ablate::checkError;

        // Copy over the options object
        PetscOptions options;
        PetscObjectGetOptions((PetscObject)originalDm, &options) >> checkError;
        PetscObjectSetOptions((PetscObject)replaceDm, options) >> checkError;
        ((DM_Plex*)(replaceDm)->data)->useHashLocation = ((DM_Plex*)originalDm->data)->useHashLocation;

        DMDestroy(&originalDm) >> checkError;
        originalDm = replaceDm;
    }
}

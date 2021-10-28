#include "dmWrapper.hpp"
#include "utilities/petscError.hpp"

static std::string getPetscObjectName(DM dm) {
    const char* name;
    PetscObjectName((PetscObject)dm) >> ablate::checkError;
    PetscObjectGetName((PetscObject)dm, &name) >> ablate::checkError;
    return std::string(name);
}

ablate::domain::DMWrapper::DMWrapper(DM dm, std::vector<std::shared_ptr<modifier::Modifier>> modifiers) : ablate::domain::Domain(getPetscObjectName(dm), modifiers) { this->dm = dm; }
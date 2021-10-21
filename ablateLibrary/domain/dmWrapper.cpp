#include "dmWrapper.hpp"
#include "utilities/petscError.hpp"

static std::string getPetscObjectName(DM dm) {
    const char* name;
    PetscObjectName((PetscObject)dm) >> ablate::checkError;
    PetscObjectGetName((PetscObject)dm, &name) >> ablate::checkError;
    return std::string(name);
}

ablate::domain::DMWrapper::DMWrapper(DM dm) : ablate::domain::Domain(getPetscObjectName(dm)) { this->dm = dm; }
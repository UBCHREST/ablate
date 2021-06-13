#include "dmWrapper.hpp"
#include "utilities/petscError.hpp"

static std::string getPetscObjectName(DM dm) {
    const char* name;
    PetscObjectName((PetscObject)dm) >> ablate::checkError;
    PetscObjectGetName((PetscObject)dm, &name) >> ablate::checkError;
    return std::string(name);
}

ablate::mesh::DMWrapper::DMWrapper(DM dm) : ablate::mesh::Mesh(getPetscObjectName(dm)) { this->dm = dm; }
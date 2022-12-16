#include "dmWrapper.hpp"

#include <utility>
#include "utilities/petscError.hpp"

static std::string getPetscObjectName(DM dm) {
    const char* name;
    PetscObjectName((PetscObject)dm) >> ablate::checkError;
    PetscObjectGetName((PetscObject)dm, &name) >> ablate::checkError;
    return {name};
}

ablate::domain::DMWrapper::DMWrapper(DM dm, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers,
                                     std::shared_ptr<parameters::Parameters> options)
    : ablate::domain::Domain(dm, getPetscObjectName(dm), std::move(fieldDescriptors), modifiers, options) {}
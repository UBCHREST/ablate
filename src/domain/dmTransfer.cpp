#include "dmTransfer.hpp"

#include <utility>
#include "utilities/petscUtilities.hpp"

static std::string getPetscObjectName(DM dm) {
    const char* name;
    PetscObjectName((PetscObject)dm) >> ablate::utilities::PetscUtilities::checkError;
    PetscObjectGetName((PetscObject)dm, &name) >> ablate::utilities::PetscUtilities::checkError;
    return {name};
}

ablate::domain::DMTransfer::DMTransfer(DM dm, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers,
                                       std::shared_ptr<parameters::Parameters> options)
    : ablate::domain::Domain(dm, getPetscObjectName(dm), std::move(fieldDescriptors), modifiers, options, false /*prevent set from options*/) {}

ablate::domain::DMTransfer::~DMTransfer() {
    if (dm) {
        DMDestroy(&dm) >> utilities::PetscUtilities::checkError;
    }
}

#include "setFromOptions.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::domain::modifiers::SetFromOptions::SetFromOptions(std::shared_ptr<parameters::Parameters> options) : petscOptions(nullptr) {
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }
}

ablate::domain::modifiers::SetFromOptions::~SetFromOptions() {
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck("SetFromOptions", &petscOptions);
    }
}

void ablate::domain::modifiers::SetFromOptions::Modify(DM &dm) {
    PetscObjectSetOptions((PetscObject)dm, petscOptions) >> checkError;
    DMSetFromOptions(dm) >> checkError;
}

#include "parser/registrar.hpp"
REGISTER_PASS_THROUGH(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::SetFromOptions, "Sets the specified options on the dm.", ablate::parameters::Parameters);
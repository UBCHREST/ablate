#include "setFromOptions.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>
ablate::domain::modifier::SetFromOptions::SetFromOptions(std::shared_ptr<parameters::Parameters> options) : petscOptions(nullptr) {
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }
}

ablate::domain::modifier::SetFromOptions::~SetFromOptions() {
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck("SetFromOptions", &petscOptions);
    }
}

void ablate::domain::modifier::SetFromOptions::Modify(DM &dm) {
    PetscObjectSetOptions((PetscObject)dm, petscOptions) >> checkError;
    DMSetFromOptions(dm) >> checkError;
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::modifier::Modifier, ablate::domain::modifier::SetFromOptions, "Sets the specified options on the dm.",
         OPT(ablate::parameters::Parameters, "options", "options used to setup the dm.  Default is global options."));
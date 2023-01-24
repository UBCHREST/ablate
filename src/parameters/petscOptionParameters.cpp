#include "petscOptionParameters.hpp"
#include "utilities/petscUtilities.hpp"
ablate::parameters::PetscOptionParameters::PetscOptionParameters(PetscOptions petscOptionsIn) : petscOptions(petscOptionsIn) {}

std::optional<std::string> ablate::parameters::PetscOptionParameters::GetString(std::string paramName) const {
    // Get the option from the petsc options database
    PetscBool found;
    char result[PETSC_MAX_OPTION_NAME];

    // add prefix to the param name
    auto paramNamePetsc = "-" + paramName;

    PetscOptionsGetString(petscOptions, NULL, paramNamePetsc.c_str(), result, PETSC_MAX_OPTION_NAME, &found) >> utilities::PetscUtilities::checkError;

    if (found) {
        return std::string(result);
    } else {
        return {};
    }
}

std::unordered_set<std::string> ablate::parameters::PetscOptionParameters::GetKeys() const {
    std::unordered_set<std::string> keys;

    // Get all of the values as a string
    char* keyString;
    PetscOptionsGetAll(petscOptions, &keyString) >> utilities::PetscUtilities::checkError;

    return keys;
}

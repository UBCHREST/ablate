#include "petscPrefixOptions.hpp"
#include <string>
#include "utilities/petscUtilities.hpp"

#define PETSC_MAX_VALUE_SIZE 2048

ablate::parameters::PetscPrefixOptions::PetscPrefixOptions(std::string prefix) : MapParameters() {
    // Create a new petsc options
    PetscOptions filteredOptions;
    PetscOptionsCreate(&filteredOptions) >> utilities::PetscUtilities::checkError;

    // March over each key
    // Get all of the values as a string
    char* keyString;
    PetscOptionsGetAll(NULL, &keyString) >> utilities::PetscUtilities::checkError;

    std::istringstream keyAndOptions(keyString);
    while (keyAndOptions) {
        std::string subString;
        keyAndOptions >> subString;

        // if this options starts with the prefix
        if (subString.find(prefix) == 0) {
            char value[PETSC_MAX_VALUE_SIZE];
            // Get the value from petsc
            PetscOptionsGetString(NULL, NULL, subString.c_str(), value, PETSC_MAX_VALUE_SIZE, NULL) >> utilities::PetscUtilities::checkError;

            // remove the prefix
            std::string name = subString.substr(prefix.size());

            // Set in the sub opt
            values[name] = std::string(value);
        }
    }

    PetscFree(keyString);
}

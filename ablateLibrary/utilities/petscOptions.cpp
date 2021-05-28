#include "petscOptions.hpp"
#include "petscError.hpp"
#include "petscsys.h"

void ablate::utilities::PetscOptionsUtils::Set(const std::string& prefix, const std::map<std::string, std::string>& options) {
    // March over and set each option in the global petsc database
    for (auto optionPair : options) {
        std::string optionName = "-" + prefix + "" + optionPair.first;
        PetscOptionsSetValue(NULL, optionName.c_str(), optionPair.second.empty() ? NULL : optionPair.second.c_str()) >> checkError;
    }
}

void ablate::utilities::PetscOptionsUtils::Set(const std::map<std::string, std::string>& options) {
    const std::string noPrefix = "";
    Set(noPrefix, options);
}

void ablate::utilities::PetscOptionsUtils::Set(PetscOptions petscOptions, const std::map<std::string, std::string>& options) {
    for (auto optionPair : options) {
        std::string optionName = "-" + optionPair.first;
        PetscOptionsSetValue(petscOptions, optionName.c_str(), optionPair.second.empty() ? NULL : optionPair.second.c_str()) >> checkError;
    }
}

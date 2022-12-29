#include "petscOptions.hpp"
#include "petscUtilities.hpp"
#include "petscsys.h"

void ablate::utilities::PetscOptionsUtils::Set(const std::string& prefix, const std::map<std::string, std::string>& options) {
    // March over and set each option in the global petsc database
    for (auto optionPair : options) {
        std::string optionName = "-" + prefix + "" + optionPair.first;
        PetscOptionsSetValue(NULL, optionName.c_str(), optionPair.second.empty() ? NULL : optionPair.second.c_str()) >> utilities::PetscUtilities::checkError;
    }
}

void ablate::utilities::PetscOptionsUtils::Set(const std::map<std::string, std::string>& options) {
    const std::string noPrefix = "";
    Set(noPrefix, options);
}

void ablate::utilities::PetscOptionsUtils::Set(PetscOptions petscOptions, const std::map<std::string, std::string>& options) {
    for (auto optionPair : options) {
        std::string optionName = "-" + optionPair.first;
        PetscOptionsSetValue(petscOptions, optionName.c_str(), optionPair.second.empty() ? NULL : optionPair.second.c_str()) >> utilities::PetscUtilities::checkError;
    }
}

void ablate::utilities::PetscOptionsDestroyAndCheck(const std::string& name, PetscOptions* options) {
    PetscInt nopt;
    PetscOptionsAllUsed(*options, &nopt) >> utilities::PetscUtilities::checkError;
    if (nopt) {
        PetscPrintf(PETSC_COMM_WORLD, "WARNING! There are options in %s you set that were not used!\n", name.c_str()) >> utilities::PetscUtilities::checkError;
        PetscPrintf(PETSC_COMM_WORLD, "WARNING! could be spelling mistake, etc!\n") >> utilities::PetscUtilities::checkError;
        if (nopt == 1) {
            PetscPrintf(PETSC_COMM_WORLD, "There is one unused database option. It is:\n") >> utilities::PetscUtilities::checkError;
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "There are %" PetscInt_FMT "unused database options. They are:\n", nopt) >> utilities::PetscUtilities::checkError;
        }
    }
    PetscOptionsLeft(*options) >> utilities::PetscUtilities::checkError;
    PetscOptionsDestroy(options) >> utilities::PetscUtilities::checkError;
}
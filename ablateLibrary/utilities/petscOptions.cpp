#include "petscOptions.hpp"
#include "petscError.hpp"
#include "petscsys.h"

void ablate::utilities::PetscOptions::Set(std::string& prefix, std::map<std::string, std::string>& options) {

    // March over and set each option in the global petsc database
    for(auto optionPair : options){
        std::string optionName = "-" + prefix + "" + optionPair.first;
        PetscOptionsSetValue(NULL, optionName.c_str(), optionPair.second.c_str()) >> checkError;
    }
}

#include "eos.hpp"
#include "generators.hpp"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::eos::EOS::EOS(std::string type, std::map<std::string, std::string> parameters) {
    // Create an eos
    EOSCreate(&eosData) >> checkError;

    // Set the type
    EOSSetType(eosData, type.c_str()) >> checkError;

    // Create the required options
    PetscOptionsCreate(&eosOptions) >> checkError;

    // Fill the options from the parameters
    utilities::PetscOptionsUtils::Set(eosOptions, parameters);
    EOSSetOptions(eosData, eosOptions) >> checkError;
    EOSSetFromOptions(eosData) >> checkError;
}

ablate::eos::EOS::~EOS() {
    if (eosData) {
        EOSDestroy(&eosData) >> checkError;
    }
    if (eosOptions) {
        PetscOptionsDestroy(&eosOptions) >> checkError;
    }
}

REGISTERDEFAULT(ablate::eos::EOS, ablate::eos::EOS, "Equation of State (EOS)", ARG(std::string, "type", "the EOS type"),
                ARG(std::map<std::string TMP_COMMA std::string>, "parameters", "EOS parameters"));
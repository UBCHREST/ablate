#include "parameters.hpp"
#include <set>
#include "environment/runEnvironment.hpp"
#include "utilities/petscUtilities.hpp"

static std::set<std::string> knownTrueValues = {"true", "TRUE", "True", "y", "Y", "Yes", "on", "ON", "On", "1", "yes", "YES"};

void ablate::parameters::Parameters::toValue(const std::string& inputString, bool& outputValue) { outputValue = knownTrueValues.count(inputString) > 0; }

void ablate::parameters::Parameters::Fill(PetscOptions options) const {
    // March over each key
    for (const auto& key : GetKeys()) {
        // prepend the name
        auto name = "-" + key;

        // Get the value
        auto value = GetString(key).value();

        // check for any environment overrides
        ablate::environment::RunEnvironment::Get().ExpandVariables(value);

        // set the options
        PetscOptionsSetValue(options, name.c_str(), value.c_str()) >> utilities::PetscUtilities::checkError;
    }
}

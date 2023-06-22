#include "factoryParameters.hpp"
#include "registrar.hpp"

ablate::parameters::FactoryParameters::FactoryParameters(std::shared_ptr<cppParser::Factory> factory) : factory(factory) {}

std::optional<std::string> ablate::parameters::FactoryParameters::GetString(std::string paramName) const {
    if (factory->Contains(paramName)) {
        auto stringArgument = cppParser::ArgumentIdentifier<std::string>{.inputName = paramName, .description = "", .optional = false};
        return factory->Get(stringArgument);
    }
    return std::optional<std::string>();
}

std::unordered_set<std::string> ablate::parameters::FactoryParameters::GetKeys() const { return factory->GetKeys(); }

REGISTER_DEFAULT_FACTORY_CONSTRUCTOR(ablate::parameters::Parameters, ablate::parameters::FactoryParameters, "Creates a parameter list based upon a factory.  Should be default for factory parsing.");
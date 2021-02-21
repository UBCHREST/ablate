#include "factoryParameters.hpp"
#include "parser/registrar.hpp"

ablate::parameters::FactoryParameters::FactoryParameters(std::shared_ptr<ablate::parser::Factory>& factory):
factory(factory)
{
}

std::optional<std::string> ablate::parameters::FactoryParameters::GetString(std::string paramName) const {
    if(factory->Contains(paramName)){
        auto stringArgument = ablate::parser::ArgumentIdentifier<std::string>{.inputName = paramName};
        return factory->Get(stringArgument);
    }
    return std::optional<std::string>();
}

REGISTER_FACTORY_CONSTRUCTOR_DEFAULT(ablate::parameters::Parameters, ablate::parameters::FactoryParameters, "Creates a parameter list based upon a factory.  Should be default for factory parsing.");
#include "mapParameters.hpp"

std::optional<std::string> ablate::parameters::MapParameters::GetString(std::string paramName) const {
    if(values.count(paramName)){
        return values.at(paramName);
    }else{
        return {};
    }
}

ablate::parameters::MapParameters::MapParameters(std::map<std::string, std::string> values)
    :values(values) {
}

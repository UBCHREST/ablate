#include "mapParameters.hpp"

ablate::parameters::MapParameters::MapParameters(std::map<std::string, std::string> values) : values(values) {}

std::optional<std::string> ablate::parameters::MapParameters::GetString(std::string paramName) const {
    if (values.count(paramName)) {
        return values.at(paramName);
    } else {
        return {};
    }
}

std::unordered_set<std::string> ablate::parameters::MapParameters::GetKeys() const {
    std::unordered_set<std::string> keys;
    for (const auto& key : values) {
        keys.insert(key.first);
    }
    return keys;
}

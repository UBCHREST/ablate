#include "mapParameters.hpp"
#include <utility>

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

ablate::parameters::MapParameters::MapParameters(std::initializer_list<std::pair<std::string, std::string>> list) {
    for (const auto& pair : list) {
        values[pair.first] = pair.second;
    }
}

std::shared_ptr<ablate::parameters::MapParameters> ablate::parameters::MapParameters::Create(std::initializer_list<std::pair<std::string, std::string>> values) {
    return std::make_shared<ablate::parameters::MapParameters>(values);
}

#include "yamlParser.hpp"

ablate::parser::YamlParser::YamlParser(const YAML::Node yamlConfiguration, std::string nodePath, std::string type) : type(type), nodePath(nodePath), yamlConfiguration(yamlConfiguration) {
    // store each child in the map with zero usages
    for (auto childNode : yamlConfiguration) {
        nodeUsages[key_to_string(childNode.first)] = 0;
    }
}

ablate::parser::YamlParser::YamlParser(std::string yamlString) : YamlParser(YAML::Load(yamlString), "root", "") {}

ablate::parser::YamlParser::YamlParser(std::filesystem::path filePath) : YamlParser(YAML::LoadFile(filePath), "root", "") {}

std::shared_ptr<ablate::parser::Factory> ablate::parser::YamlParser::GetFactory(const std::string& name) const {
    // Check to see if the child factory has already been created
    if (childFactories.count(name) == 0) {
        auto parameter = yamlConfiguration[name];
        if (!parameter) {
            throw std::invalid_argument("unable to find item " + name + " in " + nodePath);
        }

        if (!parameter.IsMap()) {
            throw std::invalid_argument("item " + name + " is expected to be a map in " + nodePath);
        }

        std::string childPath = nodePath + "/" + name;

        auto tagType = parameter.Tag();
        // Remove the ! or ? from the tag
        tagType = tagType.size() > 0 ? tagType.substr(1) : tagType;

        // mark usage and store pointer
        MarkUsage(name);
        childFactories[name] = std::shared_ptr<YamlParser>(new YamlParser(parameter, childPath, tagType));
    }

    return childFactories[name];
}

std::vector<std::shared_ptr<ablate::parser::Factory>> ablate::parser::YamlParser::GetFactorySequence(const std::string& name) const {
    auto parameter = yamlConfiguration[name];
    if (!parameter) {
        throw std::invalid_argument("unable to find list " + name + " in " + nodePath);
    }

    if (!parameter.IsSequence()) {
        throw std::invalid_argument("item " + name + " is expected to be a sequence in " + nodePath);
    }

    std::vector<std::shared_ptr<Factory>> children;

    // march over each child
    for (std::size_t i = 0; i < parameter.size(); i++) {
        std::string childName = name + "/" + std::to_string(i);

        if (childFactories.count(childName) == 0) {
            auto childParameter = parameter[i];

            if (!childParameter.IsDefined()) {
                throw std::invalid_argument("item " + childName + " is expected to be a defined in " + nodePath + "/" + name);
            }

            std::string childPath = nodePath + "/" + childName;

            auto tagType = childParameter.Tag();
            // Remove the ! or ? from the tag
            tagType = tagType.size() > 0 ? tagType.substr(1) : tagType;

            // mark usage and store pointer
            childFactories[childName] = std::shared_ptr<YamlParser>(new YamlParser(childParameter, childPath, tagType));
        }

        children.push_back(childFactories[childName]);
    }

    MarkUsage(name);

    return children;
}

std::vector<std::string> ablate::parser::YamlParser::GetUnusedValues() const {
    std::vector<std::string> unused;

    for (auto children : nodeUsages) {
        if (children.second == 0) {
            unused.push_back(nodePath + "/" + children.first);
        }
    }

    // Add any unused children from used children
    for (auto childFactory : childFactories) {
        auto unusedChildren = childFactory.second->GetUnusedValues();
        unused.insert(std::end(unused), std::begin(unusedChildren), std::end(unusedChildren));
    }

    return unused;
}

std::unordered_set<std::string> ablate::parser::YamlParser::GetKeys() const {
    std::unordered_set<std::string> keys;

    for (auto childNode : yamlConfiguration) {
        keys.insert(key_to_string(childNode.first));
    }

    return keys;
}

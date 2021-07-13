#include "yamlParser.hpp"
#include <environment/runEnvironment.hpp>
#include <utilities/fileUtility.hpp>
#include <utilities/mpiError.hpp>

ablate::parser::YamlParser::YamlParser(const YAML::Node yamlConfiguration, std::string nodePath, std::string type, bool relocateRemoteFiles)
    : type(type), nodePath(nodePath), yamlConfiguration(yamlConfiguration), relocateRemoteFiles(relocateRemoteFiles) {
    // store each child in the map with zero usages
    for (auto childNode : yamlConfiguration) {
        nodeUsages[YAML::key_to_string(childNode.first)] = 0;
    }
}

ablate::parser::YamlParser::YamlParser(std::string yamlString, bool relocateRemoteFiles) : YamlParser(YAML::Load(yamlString), "root", "", relocateRemoteFiles) {}

ablate::parser::YamlParser::YamlParser(std::filesystem::path filePath, bool relocateRemoteFiles) : YamlParser(YAML::LoadFile(filePath), "root", "", relocateRemoteFiles) {
    // add the file parent to the search directory
    searchDirectories.push_back(filePath.parent_path());
}

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
        childFactories[name] = std::shared_ptr<YamlParser>(new YamlParser(parameter, childPath, tagType, relocateRemoteFiles));
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
            childFactories[childName] = std::shared_ptr<YamlParser>(new YamlParser(childParameter, childPath, tagType, relocateRemoteFiles));
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

std::filesystem::path ablate::parser::YamlParser::Get(const ablate::parser::ArgumentIdentifier<std::filesystem::path>& identifier) const {
    // The yaml parser just refers to the global environment to file the file
    auto file = Get(ablate::parser::ArgumentIdentifier<std::string>{.inputName = identifier.inputName, .optional = identifier.optional});

    if (identifier.optional && file.empty()) {
        return {};
    } else {
        return utilities::FileUtility::LocateFile(file, MPI_COMM_WORLD, searchDirectories, relocateRemoteFiles ? environment::RunEnvironment::Get().GetOutputDirectory() : std::filesystem::path());
    }
}

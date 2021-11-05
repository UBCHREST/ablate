#include "yamlParser.hpp"
#include <environment/runEnvironment.hpp>
#include <utilities/fileUtility.hpp>
#include <utilities/mpiError.hpp>

ablate::parser::YamlParser::YamlParser(const YAML::Node yamlConfiguration, std::string nodePath, std::string type, bool relocateRemoteFiles, std::vector<std::filesystem::path> searchDirectories)
    : type(type), nodePath(nodePath), yamlConfiguration(yamlConfiguration), relocateRemoteFiles(relocateRemoteFiles), searchDirectories(searchDirectories) {
    // store each child in the map with zero usages
    for (auto childNode : yamlConfiguration) {
        nodeUsages[YAML::key_to_string(childNode.first)] = 0;
    }
}

ablate::parser::YamlParser::YamlParser(YAML::Node yamlConfiguration, bool relocateRemoteFiles, std::shared_ptr<ablate::parameters::Parameters> overwriteParameters)
    : YamlParser(yamlConfiguration, "root", "", relocateRemoteFiles) {
    // override/add any of the values in the overwriteParameters
    if (overwriteParameters) {
        for (const auto& key : overwriteParameters->GetKeys()) {
            ReplaceValue(yamlConfiguration, key, overwriteParameters->GetExpect<std::string>(key));
        }
    }
}

ablate::parser::YamlParser::YamlParser(std::string yamlString, bool relocateRemoteFiles, std::shared_ptr<ablate::parameters::Parameters> overwriteParameters)
    : YamlParser(YAML::Load(yamlString), relocateRemoteFiles, overwriteParameters) {}

ablate::parser::YamlParser::YamlParser(std::filesystem::path filePath, bool relocateRemoteFiles, std::shared_ptr<ablate::parameters::Parameters> overwriteParameters)
    : YamlParser(YAML::LoadFile(filePath), relocateRemoteFiles, overwriteParameters) {
    // add the file parent to the search directory
    searchDirectories.push_back(filePath.parent_path());
}

std::shared_ptr<ablate::parser::Factory> ablate::parser::YamlParser::GetFactory(const std::string& name) const {
    // Check to see if the child factory has already been created
    if (childFactories.count(name) == 0) {
        YAML::Node parameter;
        std::string childPath;
        std::string tagType;
        if(name.empty()){
            parameter = yamlConfiguration;
            childPath = nodePath;

            // This is the child, so assume that the tag is empty
            tagType = "";

            // Mark all children here on used, becuase they will be counted in the child
            MarkAllUsed();
        }else{
            parameter = yamlConfiguration[name];
            childPath = nodePath + "/" + name;

            if (!parameter) {
                throw std::invalid_argument("unable to find item " + name + " in " + nodePath);
            }

            tagType = parameter.Tag();
            // Remove the ! or ? from the tag
            tagType = tagType.size() > 0 ? tagType.substr(1) : tagType;
        }

        // mark usage and store pointer
        MarkUsage(name);
        childFactories[name] = std::shared_ptr<YamlParser>(new YamlParser(parameter, childPath, tagType, relocateRemoteFiles, searchDirectories));
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
            childFactories[childName] = std::shared_ptr<YamlParser>(new YamlParser(childParameter, childPath, tagType, relocateRemoteFiles, searchDirectories));
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

void ablate::parser::YamlParser::Print(std::ostream& stream) const { stream << "---" << std::endl << yamlConfiguration << std::endl; }

void ablate::parser::YamlParser::ReplaceValue(YAML::Node& yamlConfiguration, std::string key, std::string value) {
    // Check to see if there are any separators in the key
    auto separator = key.find("::");
    if (separator == std::string::npos) {
        // we are at the end, so just apply it
        yamlConfiguration[key] = value;
    } else {
        // strip off the first part of the key and try again
        auto thisKey = key.substr(0, separator);

        // check to see if this key is an index
        if (!thisKey.empty() && thisKey[0] == '[' && thisKey.back() == ']') {
            auto index = std::stoi(thisKey.substr(1, thisKey.size() - 2));
            auto childConfig = yamlConfiguration[index];
            ReplaceValue(childConfig, key.substr(separator + 2), value);
        } else {
            auto childConfig = yamlConfiguration[thisKey];
            ReplaceValue(childConfig, key.substr(separator + 2), value);
        }
    }
}

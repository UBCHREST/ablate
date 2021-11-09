#ifndef ABLATELIBRARY_YAMLPARSER_HPP
#define ABLATELIBRARY_YAMLPARSER_HPP

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <iostream>
#include "factory.hpp"
#include "parameters/parameters.hpp"

namespace ablate::parser {
class YamlParser : public Factory {
   private:
    const std::string type;
    const std::string nodePath;
    const YAML::Node yamlConfiguration;
    const bool relocateRemoteFiles;
    mutable std::map<std::string, int> nodeUsages;
    mutable std::map<std::string, std::shared_ptr<YamlParser>> childFactories;

    // store a list of local directories to search for files
    std::vector<std::filesystem::path> searchDirectories;

    /***
     * private constructor to create a sub factory
     * @param yamlConfiguration
     * @param nodePath
     * @param type
     */
    YamlParser(const YAML::Node yamlConfiguration, std::string nodePath, std::string type, bool relocateRemoteFiles, std::vector<std::filesystem::path> searchDirectories = {});
    inline void MarkUsage(const std::string& key) const {
        if (!key.empty()) {
            nodeUsages[key]++;
        }
    }

    /**
     * Marks all of the keys used.
     */
    inline void MarkAllUsed() const {
        for (auto& pairs : nodeUsages) {
            pairs.second++;
        }
    }

    /***
     * Helper Function to get the correct parameters
     */
    template <typename T>
    inline YAML::Node GetParameter(const ArgumentIdentifier<T>& identifier) const {
        // treat this yamlConfiguration as the item if the identifier is default
        if (identifier.inputName.empty()) {
            return yamlConfiguration;
        } else if (yamlConfiguration.IsMap()) {
            return yamlConfiguration[identifier.inputName];
        } else {
            return YAML::Node(YAML::NodeType::Undefined);
        }
    }

    template <typename T>
    inline T GetValueFromYaml(const ArgumentIdentifier<T>& identifier) const {
        // treat this yamlConfiguration as the item if the identifier is default
        auto parameter = GetParameter(identifier);
        if (!parameter) {
            if (identifier.optional) {
                return {};
            } else {
                throw std::invalid_argument("unable to locate " + identifier.inputName + " in " + nodePath);
            }
        }
        MarkUsage(identifier.inputName);
        return parameter.template as<T>();
    }

    /**
     * recursive call to update parameters
     */
    static void ReplaceValue(YAML::Node& yamlConfiguration, std::string key, std::string value);

   public:
    explicit YamlParser(const YAML::Node yamlConfiguration, bool relocateRemoteFiles = false, std::shared_ptr<parameters::Parameters> overwriteParameters = {});
    ~YamlParser() override = default;

    /***
     * Direct creation using a yaml string
     * @param yamlString
     */
    explicit YamlParser(std::string yamlString, bool relocateRemoteFiles = false, std::shared_ptr<parameters::Parameters> overwriteParameters = {});

    /***
     * Read in file from system
     * @param filePath
     */
    explicit YamlParser(std::filesystem::path filePath, bool relocateRemoteFiles = true, std::shared_ptr<parameters::Parameters> overwriteParameters = {});

    /* gets the class type represented by this factory */
    const std::string& GetClassType() const override { return type; }

    /* return a string*/
    std::string Get(const ArgumentIdentifier<std::string>& identifier) const override {
        // treat this yamlConfiguration as the item if the identifier is default
        auto parameter = GetParameter(identifier);
        if (!parameter) {
            if (identifier.optional) {
                return {};
            } else {
                throw std::invalid_argument("unable to locate " + identifier.inputName + " in " + nodePath);
            }
        }
        MarkUsage(identifier.inputName);
        if (parameter.IsSequence()) {
            // Merge the results into a single space seperated string
            std::stringstream ss;
            for (const auto& v : parameter) {
                ss << v.template as<std::string>() << " ";
            }
            return ss.str();
        } else {
            return parameter.template as<std::string>();
        }
    }

    std::vector<std::string> Get(const ArgumentIdentifier<std::vector<std::string>>& identifier) const override { return GetValueFromYaml<std::vector<std::string>>(identifier); }

    bool Get(const ArgumentIdentifier<bool>& identifier) const override { return GetValueFromYaml<bool>(identifier); };

    double Get(const ArgumentIdentifier<double>& identifier) const override { return GetValueFromYaml<double>(identifier); };

    std::vector<int> Get(const ArgumentIdentifier<std::vector<int>>& identifier) const override { return GetValueFromYaml<std::vector<int>>(identifier); }

    std::vector<double> Get(const ArgumentIdentifier<std::vector<double>>& identifier) const override { return GetValueFromYaml<std::vector<double>>(identifier); }

    std::map<std::string, std::string> Get(const ArgumentIdentifier<std::map<std::string, std::string>>& identifier) const override {
        return GetValueFromYaml<std::map<std::string, std::string>>(identifier);
    }

    /* return an int for the specified identifier*/
    int Get(const ArgumentIdentifier<int>& identifier) const override { return GetValueFromYaml<int>(identifier); }

    /* return the path to file specified */
    std::filesystem::path Get(const ArgumentIdentifier<std::filesystem::path>& identifier) const override;

    /* return a factory that serves as the root of the requested item */
    std::shared_ptr<Factory> GetFactory(const std::string& name) const override;

    /* get all children as factory */
    std::vector<std::shared_ptr<Factory>> GetFactorySequence(const std::string& name) const override;

    bool Contains(const std::string& name) const override { return yamlConfiguration.IsMap() ? yamlConfiguration[name] != nullptr : false; };

    std::unordered_set<std::string> GetKeys() const override;

    /** get unused values **/
    std::vector<std::string> GetUnusedValues() const;

    /** print a copy of the yaml input as updated **/
    void Print(std::ostream&) const;
};
}  // namespace ablate::parser

#endif  // ABLATELIBRARY_YAMLPARSER_HPP

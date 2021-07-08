#ifndef ABLATELIBRARY_YAMLPARSER_HPP
#define ABLATELIBRARY_YAMLPARSER_HPP

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include "factory.hpp"

namespace ablate::parser {
class YamlParser : public Factory {
   private:
    const std::string type;
    const std::string nodePath;
    const YAML::Node yamlConfiguration;
    mutable std::map<std::string, int> nodeUsages;
    mutable std::map<std::string, std::shared_ptr<YamlParser>> childFactories;

    /***
     * private constructor to create a sub factory
     * @param yamlConfiguration
     * @param nodePath
     * @param type
     */
    YamlParser(const YAML::Node yamlConfiguration, std::string nodePath, std::string type);
    inline void MarkUsage(const std::string& key) const { nodeUsages[key]++; }

    template <typename T>
    inline T GetValueFromYaml(const ArgumentIdentifier<T>& identifier) const {
        auto parameter = yamlConfiguration[identifier.inputName];
        if (!parameter) {
            if (identifier.optional) {
                return {};
            } else {
                throw std::invalid_argument("unable to " + identifier.inputName + " in " + nodePath);
            }
        }
        MarkUsage(identifier.inputName);
        return parameter.template as<T>();
    }

   public:
    /***
     * Direct creation using a yaml string
     * @param yamlString
     */
    explicit YamlParser(std::string yamlString);
    ~YamlParser() override = default;

    /***
     * Read in file from system
     * @param filePath
     */
    explicit YamlParser(std::filesystem::path filePath);

    /* gets the class type represented by this factory */
    const std::string& GetClassType() const override { return type; }

    /* return a string*/
    std::string Get(const ArgumentIdentifier<std::string>& identifier) const override {
        auto parameter = yamlConfiguration[identifier.inputName];
        if (!parameter) {
            if (identifier.optional) {
                return {};
            } else {
                throw std::invalid_argument("unable to " + identifier.inputName + " in " + nodePath);
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

    /* return a factory that serves as the root of the requested item */
    std::shared_ptr<Factory> GetFactory(const std::string& name) const override;

    /* get all children as factory */
    std::vector<std::shared_ptr<Factory>> GetFactorySequence(const std::string& name) const override;

    bool Contains(const std::string& name) const override { return yamlConfiguration[name] != nullptr; };

    std::unordered_set<std::string> GetKeys() const override;

    /** get unused values **/
    std::vector<std::string> GetUnusedValues() const;
};
}  // namespace ablate::parser

#endif  // ABLATELIBRARY_YAMLPARSER_HPP

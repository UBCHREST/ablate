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

    inline void MarkUsage(const std::string& key) const{
        nodeUsages[key] ++;
    }

   public:
    /***
     * Direct creation using a yaml string
     * @param yamlString
     */
    explicit YamlParser(std::string yamlString);

    /***
     * Read in file from system
     * @param filePath
     */
    explicit YamlParser(std::filesystem::path filePath);

    /* gets the class type represented by this factory */
    const std::string& GetClassType() const override{
        return type;
    }

    /* return a string*/
    std::string Get(const ArgumentIdentifier<std::string>& identifier) const override;

    /* and a list of strings */
    std::vector<std::string>  Get(const ArgumentIdentifier<std::vector<std::string>>& identifier) const override;

    std::map<std::string, std::string> Get(const ArgumentIdentifier<std::map<std::string, std::string>>& identifier) const override;

    /* return an int for the specified identifier*/
    int Get(const ArgumentIdentifier<int>& identifier) const override;

    /* return a factory that serves as the root of the requested item */
    std::shared_ptr<Factory> GetFactory(const std::string& name) const override;

    /* get all children as factory */
    std::vector<std::shared_ptr<Factory>> GetFactorySequence(const std::string& name) const override;

    bool Contains(const std::string& name) const override{
        return yamlConfiguration[name] != nullptr;
    };

    /** get unused values **/
    std::vector<std::string> GetUnusedValues() const;
};
}  // namespace ablate::parser

#endif  // ABLATELIBRARY_YAMLPARSER_HPP

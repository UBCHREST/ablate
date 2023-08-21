#ifndef ABLATELIBRARY_MAPPARAMETERS_HPP
#define ABLATELIBRARY_MAPPARAMETERS_HPP

#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include "parameters.hpp"

namespace ablate::parameters {
class MapParameters : public Parameters {
   protected:
    std::map<std::string, std::string> values;

   public:
    /**
     * Helper class to simplify MapParameters initializer_list
     */
    struct Parameter {
        template <typename T>
        Parameter(std::string_view key, T value) : key{key} {
            // convert to string
            std::stringstream ss;
            ss << value;
            ss >> this->value;
        }

        std::string key;
        std::string value;
    };

    /**
     * Takes a list of parameters
     */
    MapParameters(std::initializer_list<Parameter>);

    /*
     * Take a map directly
     */
    explicit MapParameters(std::map<std::string, std::string> values = {});

    /**
     * Gets string version of parameter
     * @param paramName
     * @return
     */
    [[nodiscard]] std::optional<std::string> GetString(std::string paramName) const override;

    /**
     * List all keys in the domain
     * @return
     */
    [[nodiscard]] std::unordered_set<std::string> GetKeys() const override;

    /**
     * Provides raw access to the map
     * @return
     */
    [[nodiscard]] const std::map<std::string, std::string>& GetMap() const { return values; }

    /**
     * Allow inserting additional items into a map
     * @tparam T
     * @param key
     * @param value
     */
    template <class T>
    void Insert(std::string key, T value) {
        // convert to string
        std::stringstream ss;
        ss << value;
        values[key] = ss.str();
    }

    /**
     * static helper function to create a new MapParameters shared pointer from a list of parameters
     * ablate::parameters::MapParameters::Create({{"item1", "value1"}, {"item2", "value2"}, {"item3", 234}});
     * @return
     */
    static std::shared_ptr<MapParameters> Create(std::initializer_list<Parameter>);

    /**
     * static helper function to create a new MapParameters shared pointer from a map of <string, string>
     * @return
     */
    static std::shared_ptr<MapParameters> Create(const std::map<std::string, std::string>& values);
};
}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_MAPPARAMETERS_HPP

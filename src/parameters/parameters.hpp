#ifndef ABLATELIBRARY_PARAMETERS_HPP
#define ABLATELIBRARY_PARAMETERS_HPP
#include <petsc.h>
#include <array>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "parameterException.hpp"

namespace ablate::parameters {

class Parameters {
   private:
    template <typename T>
    static void toValue(const std::string& inputString, T& outputValue) {
        std::istringstream ss(inputString);
        ss >> outputValue;
    }

    template <typename T>
    static void toValue(const std::string& inputString, std::vector<T>& outputValue) {
        std::istringstream ss(inputString);
        T tempValue;
        while (ss >> tempValue) {
            outputValue.push_back(tempValue);
        }
    }

    template <typename T, std::size_t N>
    static void toValue(const std::string& inputString, std::array<T, N>& outputValue) {
        std::istringstream ss(inputString);
        T tempValue;
        std::size_t index = 0;
        // set to default value of T
        T defaultValue = {};
        outputValue.fill(defaultValue);

        while (ss >> tempValue && index < N) {
            outputValue[index] = tempValue;
            index++;
        }
    }

    // value specific cases
    static void toValue(const std::string& inputString, bool& outputValue);

   public:
    virtual ~Parameters() = default;

    virtual std::optional<std::string> GetString(std::string paramName) const = 0;
    virtual std::unordered_set<std::string> GetKeys() const = 0;

    template <typename T>
    std::optional<T> Get(std::string paramName) const {
        auto value = GetString(paramName);
        if (value.has_value()) {
            T num;
            toValue(value.value(), num);
            return num;
        } else {
            return {};
        }
    }

    template <typename T>
    T Get(std::string paramName, T defaultValue) const {
        auto value = GetString(paramName);
        if (value.has_value()) {
            T num;
            toValue(value.value(), num);
            return num;
        } else {
            return defaultValue;
        }
    }

    template <typename T>
    T GetExpect(std::string paramName) const {
        auto value = GetString(paramName);
        if (value.has_value()) {
            T num;
            toValue(value.value(), num);
            return num;
        } else {
            throw ParameterException(paramName);
        }
    }

    /**
     * tries to convert each item in this parameter to T and places in map
     * @tparam T
     * @param paramName
     * @return
     */
    template <typename T>
    std::map<std::string, T> ToMap() const {
        std::map<std::string, T> map;
        for (const auto& key : GetKeys()) {
            map[key] = GetExpect<T>(key);
        }
        return map;
    }

    void Fill(PetscOptions options) const;

    template <typename T>
    void Fill(int numberValues, const char* const* valueNames, T* constantArray) const {
        // March over each parameter
        for (int n = 0; n < numberValues; n++) {
            // make a temp string
            auto stringName = std::string(valueNames[n]);

            // Set the value
            constantArray[n] = GetExpect<T>(stringName);
        }
    }

    template <typename T>
    void Fill(int numberValues, const char* const* valueNames, T* constantArray, std::map<std::string, T> defaultValues) const {
        // March over each parameter
        for (int n = 0; n < numberValues; n++) {
            // make a temp string
            auto stringName = std::string(valueNames[n]);

            // Set the value
            if (defaultValues.count(stringName)) {
                constantArray[n] = Get<T>(stringName, defaultValues[stringName]);
            } else {
                // no default value
                constantArray[n] = GetExpect<T>(stringName);
            }
        }
    }
};
}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_PARAMETERS_HPP

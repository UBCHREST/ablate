#ifndef ABLATELIBRARY_PARAMETERS_HPP
#define ABLATELIBRARY_PARAMETERS_HPP
#include <petsc.h>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include "parameterException.hpp"

namespace ablate::parameters {

class Parameters {
   private:
    template <typename T>
    static void toValue(const std::string& inputString, T& outputValue) {
        std::istringstream ss(inputString);
        ss >> outputValue;
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
            if(defaultValues.count(stringName)){
                constantArray[n] = Get<T>(stringName, defaultValues[stringName]);
            }else{
                // no default value
                constantArray[n] = GetExpect<T>(stringName);
            }
        }
    }
};
}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_PARAMETERS_HPP

#ifndef ABLATELIBRARY_PARAMETERS_HPP
#define ABLATELIBRARY_PARAMETERS_HPP
#include <optional>
#include <sstream>
#include <string>
#include "parameterException.hpp"

namespace ablate::parameters {

class Parameters {
   public:
    virtual ~Parameters() = default;

    virtual std::optional<std::string> GetString(std::string paramName) const = 0;

    template <typename T>
    std::optional<T> Get(std::string paramName) const {
        auto value = GetString(paramName);
        if (value.has_value()) {
            std::istringstream ss(value.value());
            T num;
            ss >> num;
            return num;
        } else {
            return {};
        }
    }

    template <typename T>
    T Get(std::string paramName, T defaultValue) const {
        auto value = GetString(paramName);
        if (value.has_value()) {
            std::istringstream ss(value.value());
            T num;
            ss >> num;
            return num;
        } else {
            return defaultValue;
        }
    }

    template <typename T>
    T GetExpect(std::string paramName) const {
        auto value = GetString(paramName);
        if (value.has_value()) {
            std::istringstream ss(value.value());
            T num;
            ss >> num;
            return num;
        } else {
            throw ParameterException(paramName);
        }
    }

    template <typename T>
    void Fill(int numberValues, const char *const *valueNames, T *constantArray) const {
        // March over each parameter
        for (int n = 0; n < numberValues; n++) {
            // make a temp string
            auto stringName = std::string(valueNames[n]);

            // Set the value
            constantArray[n] = GetExpect<T>(stringName);
        }
    }
};
}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_PARAMETERS_HPP

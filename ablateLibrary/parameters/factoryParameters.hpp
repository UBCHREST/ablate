#ifndef ABLATELIBRARY_FACTORYPARAMETERS_HPP
#define ABLATELIBRARY_FACTORYPARAMETERS_HPP
#include <optional>
#include <sstream>
#include <string>
#include "parameterException.hpp"
#include "parameters.hpp"
#include "parser/factory.hpp"

namespace ablate::parameters {

class FactoryParameters : public Parameters {
   private:
    std::shared_ptr<ablate::parser::Factory> factory;

   public:
    explicit FactoryParameters(std::shared_ptr<ablate::parser::Factory> factory);

    std::optional<std::string> GetString(std::string paramName) const override;

    virtual std::unordered_set<std::string> GetKeys() const override;
};
}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_FACTORYPARAMETERS_HPP

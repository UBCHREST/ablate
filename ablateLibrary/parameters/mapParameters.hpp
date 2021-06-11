#ifndef ABLATELIBRARY_MAPPARAMETERS_HPP
#define ABLATELIBRARY_MAPPARAMETERS_HPP

#include <map>
#include "parameters.hpp"
#include <memory>
#include <iostream>
namespace ablate::parameters {
class MapParameters : public Parameters {
   protected:
    std::map<std::string, std::string> values;

   public:
    MapParameters(std::map<std::string, std::string> values = {});
    std::optional<std::string> GetString(std::string paramName) const override;
    std::unordered_set<std::string> GetKeys() const override;
};
}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_MAPPARAMETERS_HPP

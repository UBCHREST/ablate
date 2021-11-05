#ifndef ABLATELIBRARY_MAPPARAMETERS_HPP
#define ABLATELIBRARY_MAPPARAMETERS_HPP

#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include "parameters.hpp"

namespace ablate::parameters {
class MapParameters : public Parameters {
   protected:
    std::map<std::string, std::string> values;

   public:
    MapParameters(std::initializer_list<std::pair<std::string, std::string>>);
    MapParameters(std::map<std::string, std::string> values = {});
    std::optional<std::string> GetString(std::string paramName) const override;
    std::unordered_set<std::string> GetKeys() const override;


    static std::shared_ptr<MapParameters> Create(std::initializer_list<std::pair<std::string, std::string>>);
};
}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_MAPPARAMETERS_HPP

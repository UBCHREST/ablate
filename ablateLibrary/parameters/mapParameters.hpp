#ifndef ABLATELIBRARY_MAPPARAMETERS_HPP
#define ABLATELIBRARY_MAPPARAMETERS_HPP

#include "parameters.hpp"
#include <map>

namespace ablate {
namespace parameters {
class MapParameters : public Parameters {
   private:
    const std::map<std::string, std::string> values;

   public:
    MapParameters(std::map<std::string, std::string> values);
    std::optional<std::string> GetString(std::string paramName) const override;
};
}
}
#endif  // ABLATELIBRARY_MAPPARAMETERS_HPP

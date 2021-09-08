#ifndef ABLATELIBRARY_RESTORESTATE_HPP
#define ABLATELIBRARY_RESTORESTATE_HPP

#include "parameters/parameters.hpp"
#include <string>

namespace ablate::environment {

class RestoreState: public ablate::parameters::Parameters {
   public:
    virtual std::optional<std::string> GetString(std::string paramName) const override = 0;
    virtual std::unordered_set<std::string> GetKeys() const override = 0;
    virtual void Get(const std::string&, Vec) const = 0;
};
}  // namespace ablate::environment
#endif  // ABLATELIBRARY_SAVESTATE_HPP

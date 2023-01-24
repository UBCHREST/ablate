#ifndef ABLATELIBRARY_EMPTY_PARAMETERS_HPP
#define ABLATELIBRARY_EMPTY_PARAMETERS_HPP

#include <memory>
#include <optional>
#include <set>
#include "parameters.hpp"

namespace ablate::parameters {
/**
 * Simple empty set of parameters
 */
class EmptyParameters : public Parameters {
   public:
    std::optional<std::string> GetString(std::string paramName) const override { return {}; }

    std::unordered_set<std::string> GetKeys() const override { return {}; }

    /**
     * Creates an empty set of parameters
     * @return
     */
    inline static std::shared_ptr<EmptyParameters> Create() { return std::make_shared<EmptyParameters>(); }

    /**
     * Checks for null, if null returns an empty set o f parameters
     * @return
     */
    inline static std::shared_ptr<Parameters> Check(std::shared_ptr<Parameters> in) { return in ? in : Create(); }
};
}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_EMPTY_PARAMETERS_HPP

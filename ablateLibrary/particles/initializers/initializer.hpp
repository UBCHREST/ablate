#ifndef ABLATELIBRARY_INITIALIZER_HPP
#define ABLATELIBRARY_INITIALIZER_HPP
#include <string>
#include <map>
#include "flow/flow.hpp"

namespace ablate::particles::initializers {
class Initializer {
   protected:
    std::map<std::string, std::string> arguments;
   public:
    Initializer(std::map<std::string, std::string> arguments);
    virtual ~Initializer() = default;

    virtual void Initialize(ablate::flow::Flow& flow, DM particleDM) = 0;
};
}

#endif  // ABLATELIBRARY_INITIALIZER_HPP

#ifndef ABLATELIBRARY_INITIALIZER_HPP
#define ABLATELIBRARY_INITIALIZER_HPP
#include <map>
#include <string>
#include "flow/flow.hpp"

namespace ablate::particles::initializers {
class Initializer {
   protected:
    PetscOptions petscOptions;

   public:
    Initializer() = default;
    virtual ~Initializer() =default;

    virtual void Initialize(ablate::flow::Flow& flow, DM particleDM) = 0;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_INITIALIZER_HPP

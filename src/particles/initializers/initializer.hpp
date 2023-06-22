#ifndef ABLATELIBRARY_PARTICLE_INITIALIZER_HPP
#define ABLATELIBRARY_PARTICLE_INITIALIZER_HPP
#include <petsc.h>
#include <map>
#include <string>
#include "solver/solver.hpp"

namespace ablate::particles::initializers {
class Initializer {
   protected:
    PetscOptions petscOptions;

   public:
    Initializer() = default;
    virtual ~Initializer() = default;

    virtual void Initialize(ablate::domain::SubDomain& flow, DM particleDM) = 0;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_INITIALIZER_HPP

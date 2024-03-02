#ifndef ABLATELIBRARY_PARTICLE_INITIALIZER_HPP
#define ABLATELIBRARY_PARTICLE_INITIALIZER_HPP
#include <petsc.h>
#include <map>
#include <string>
#include "solver/solver.hpp"

namespace ablate::particles::initializers {
class Initializer {
   public:
    /**
     * Interface for initializing particles in the particle solver.
     */
    Initializer() = default;
    virtual ~Initializer() = default;

    /**
     * Initialize method to insert particles in the particle DM
     * @param flow the flow/background mesh
     * @param particleDM the particle dm to hold the new particles
     */
    virtual void Initialize(ablate::domain::SubDomain& flow, DM particleDM) = 0;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_PARTICLE_INITIALIZER_HPP

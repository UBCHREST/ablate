#ifndef ABLATELIBRARY_PARTICLE_PROCESS_HPP
#define ABLATELIBRARY_PARTICLE_PROCESS_HPP

#include "particles/accessors/rhsAccessor.hpp"
#include "particles/accessors/swarmAccessor.hpp"
#include "particles/accessors/eulerianAccessor.hpp"

namespace ablate::particles::processes {

class Process {
   public:
    virtual ~Process() = default;

    /**
     * virtual function to compute the rhs of the governing equations for the particles
     * @param swarmData
     * @param rhsData
     */
    virtual void ComputeRHS(PetscReal time, accessors::SwarmAccessor& swarmAccessor, accessors::RhsAccessor& rhsAccessor, accessors::EulerianAccessor& eulerianAccessor) = 0;
};

}  // namespace ablate::particles::processes

#endif  // ABLATELIBRARY_PROCESS_HPP

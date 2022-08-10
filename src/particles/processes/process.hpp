#ifndef ABLATELIBRARY_PARTICLE_PROCESS_HPP
#define ABLATELIBRARY_PARTICLE_PROCESS_HPP

#include "particles/rhsData.hpp"
#include "particles/swarmData.hpp"

namespace ablate::particles::processes {

class Process {
   public:
    virtual ~Process() = default;

    /**
     * virtual function to compute the rhs of the governing equations for the particles
     * @param swarmData
     * @param rhsData
     */
    virtual void ComputeRHS(PetscInt np, PetscReal time, const SwarmData& swarmData, const RhsData& rhsData){};
};

}  // namespace ablate::particles::processes

#endif  // ABLATELIBRARY_PROCESS_HPP

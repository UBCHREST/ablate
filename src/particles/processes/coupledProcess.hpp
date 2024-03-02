#ifndef ABLATELIBRARY_PARTICLE_COUPLEDPROCESS_HPP
#define ABLATELIBRARY_PARTICLE_COUPLEDPROCESS_HPP

#include "particles/accessors/eulerianSourceAccessor.hpp"
#include "particles/processes/process.hpp"

namespace ablate::particles::processes {

/**
 * interface used for coupled processes to compute the eulerian source terms
 */
class CoupledProcess : public ablate::particles::processes::Process {
   public:
    /**
     * virtual function for computing the eulerian source terms given
     * @param startTime the start of the time step
     * @param endTime the end of the time steo
     * @param swarmAccessorPreStep the particle values before the time step
     * @param swarmAccessorPostStep the particle values after the time step
     * @param eulerianSourceAccessor the available eulerian source term
     */
    virtual void ComputeEulerianSource(PetscReal startTime, PetscReal endTime, accessors::SwarmAccessor& swarmAccessorPreStep, accessors::SwarmAccessor& swarmAccessorPostStep,
                                       accessors::EulerianSourceAccessor& eulerianSourceAccessor) = 0;
};

}  // namespace ablate::particles::processes

#endif  // ABLATELIBRARY_PARTICLE_COUPLEDPROCESS_HPP

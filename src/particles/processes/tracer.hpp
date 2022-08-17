#ifndef ABLATELIBRARY_TRACER_HPP
#define ABLATELIBRARY_TRACER_HPP

#include "process.hpp"

namespace ablate::particles::processes {

class Tracer : public Process {
   private:
    const std::string eulerianVelocityField;

   public:
    /**
     * Advects the particles with the flow velocity
     * @param eulerianVelocityField
     */
    explicit Tracer(const std::string& eulerianVelocityField = {});

    /**
     * computes the source terms to integrate the particle location with the flow velocity
     * @param time
     * @param swarmAccessor
     * @param rhsAccessor
     * @param eulerianAccessor
     */
    void ComputeRHS(PetscReal time, accessors::SwarmAccessor& swarmAccessor, accessors::RhsAccessor& rhsAccessor, accessors::EulerianAccessor& eulerianAccessor) override;
};

}  // namespace ablate::particles::processes
#endif  // ABLATELIBRARY_TRACER_HPP

#ifndef ABLATELIBRARY_INERTIAL_HPP
#define ABLATELIBRARY_INERTIAL_HPP

#include <array>
#include "process.hpp"

namespace ablate::particles::processes {

class Inertial : public Process {
   private:
    //!  fluid density needed for particle drag force
    const PetscReal fluidDensity;

    //! fluid viscosity needed for particle drag force
    const PetscReal fluidViscosity;

    //! gravity field
    const std::array<PetscReal, 3> gravityField;

    //! the location of the velocity field
    const std::string eulerianVelocityField;

   public:
    /**
     * Advects the particles with the flow velocity based upon a drag law
     * @param parameters input parameters for the velocity field
     * @param eulerianVelocityField optional field where eulerian velocity is defined (default "velocity")
     */
    explicit Inertial(const std::shared_ptr<parameters::Parameters>& parameters, const std::string& eulerianVelocityField = {});

    /**
     * computes the source terms to integrate the particle location with the flow velocity with drag
     * @param time
     * @param swarmAccessor
     * @param rhsAccessor
     * @param eulerianAccessor
     */
    void ComputeRHS(PetscReal time, accessors::SwarmAccessor& swarmAccessor, accessors::RhsAccessor& rhsAccessor, accessors::EulerianAccessor& eulerianAccessor) override;
};

}  // namespace ablate::particles::processes
#endif  // ABLATELIBRARY_INERTIAL_HPP

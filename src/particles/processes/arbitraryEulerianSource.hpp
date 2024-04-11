#ifndef ABLATELIBRARY_ARBITRARYEULERIANSOURCE_HPP
#define ABLATELIBRARY_ARBITRARYEULERIANSOURCE_HPP

#include <string>
#include "coupledProcess.hpp"

namespace ablate::particles::processes {

class ArbitraryEulerianSource : public CoupledProcess {
   private:
    //! store the coupled field name
    const std::string coupledFieldName;

    //! the function for the field, it should be the same size as the field.  This function is integrated in time
    const std::shared_ptr<mathFunctions::MathFunction> sourceFunction;

   public:
    /**
     * Adds an arbitrary source function for each particle to the eulerian field
     * @param coupledFieldName the name of the eulerian coupled field
     * @param sourceFunction the function to compute the source
     */
    ArbitraryEulerianSource(std::string coupledFieldName, std::shared_ptr<mathFunctions::MathFunction> sourceFunction);

    /**
     * There is no RHS function for the ArbitraryEulerianSource
     */
    void ComputeRHS(PetscReal time, accessors::SwarmAccessor& swarmAccessor, accessors::RhsAccessor& rhsAccessor, accessors::EulerianAccessor& eulerianAccessor) override {}

    /**
     * Add the arbitrary source to the eulerianSourceAccessor
     * @param startTime
     * @param endTime
     * @param swarmAccessorPreStep
     * @param swarmAccessorPostStep
     * @param eulerianSourceAccessor
     */
    void ComputeEulerianSource(PetscReal startTime, PetscReal endTime, accessors::SwarmAccessor& swarmAccessorPreStep, accessors::SwarmAccessor& swarmAccessorPostStep,
                               accessors::EulerianSourceAccessor& eulerianSourceAccessor) override;
};

}  // namespace ablate::particles::processes
#endif  // ABLATELIBRARY_INERTIAL_HPP

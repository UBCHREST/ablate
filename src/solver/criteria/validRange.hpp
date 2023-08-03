#ifndef ABLATELIBRARY_VALIDRANGE_HPP
#define ABLATELIBRARY_VALIDRANGE_HPP

#include <monitors/logs/log.hpp>
#include <utilities/mathUtilities.hpp>
#include "convergenceCriteria.hpp"

namespace ablate::solver::criteria {
/**
 * This class throws a convergence exception if there are no values in the valid range
 */
class ValidRange : public ConvergenceCriteria {
   private:
    //! hold the variable name to check
    const std::string variableName;

    //! hold the upper and lower bounds
    const double lowerBound;
    const double upperBound;

    //! hold the region to check for this variable
    const std::shared_ptr<ablate::domain::Region> region;

   public:
    /**
     * Create a new variable bound check that throws an exception if there are no values in the valid range
     * @param variableName
     * @param lowerBound values lower than this will throw an exception
     * @param upperBound values larger than this will throw an exception
     * @param region the region for which to apply this check
     */
    explicit ValidRange(std::string variableName, double lowerBound, double upperBound, std::shared_ptr<ablate::domain::Region> region);

    /**
     * This does not need to Initialize
     * @param domain
     */
    void Initialize(const ablate::domain::Domain& domain) override {}

    /**
     * Check for convergence and then update the previous state
     * @param domain
     * @param time the current simulation time
     * @param step the current step in the solution
     * @param optional log that can be used to report convergence status
     */
    bool CheckConvergence(const ablate::domain::Domain& domain, PetscReal time, PetscInt step, const std::shared_ptr<ablate::monitors::logs::Log>& log) override;
};

}  // namespace ablate::solver::criteria

#endif  // ABLATELIBRARY_VALIDRANGE_HPP

#ifndef ABLATELIBRARY_VARIABLECHANGE_HPP
#define ABLATELIBRARY_VARIABLECHANGE_HPP

#include <monitors/logs/log.hpp>
#include <utilities/mathUtilities.hpp>
#include "convergenceCriteria.hpp"

namespace ablate::solver::criteria {
/**
 * This class checks for a relative change in the specified variable between checks
 */
class VariableChange : public ConvergenceCriteria {
   private:
    //! hold the variable name to check
    const std::string variableName;

    //! hold the convergence tolerance for the l2 norm
    const PetscReal convergenceTolerance;

    //! the norm type to use when computing tolerance
    const ablate::utilities::MathUtilities::Norm convergenceNorm;

    //! hold the region to check for this variable
    const std::shared_ptr<ablate::domain::Region> region;

    // Store the previous value
    Vec previousValues = nullptr;

   public:
    /**
     * Create a new variable check
     * @param variableName
     * @param convergenceTolerance the tolerance for checking convergence
     * @param convergenceNorm the norm type for checking convergence
     * @param region
     */
    explicit VariableChange(std::string variableName, double convergenceTolerance, ablate::utilities::MathUtilities::Norm convergenceNorm, std::shared_ptr<ablate::domain::Region> region);

    /**
     * Clean up the petsc values
     */
    ~VariableChange() override;

    /**
     * Setup a vector to hold the required variable
     * @param domain
     */
    void Initialize(const ablate::domain::Domain& domain) override;

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

#endif  // ABLATELIBRARY_VARIABLECHANGE_HPP

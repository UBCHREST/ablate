#ifndef ABLATELIBRARY_CONVERGENCECRITERIA_HPP
#define ABLATELIBRARY_CONVERGENCECRITERIA_HPP

#include <domain/domain.hpp>
#include <utilities/nonCopyable.hpp>
namespace ablate::solver::criteria {

/**
 * This is an interface for convergence checkers.
 */
class ConvergenceCriteria : private ablate::utilities::NonCopyable {
   public:
    // Allow criteria cleanup
    virtual ~ConvergenceCriteria() = default;

    /**
     * Setup any required memory storage for the domain
     * @param domain
     */
    virtual void Initialize(const ablate::domain::Domain& domain) = 0;

    /**
     * Check to see if there is convergence at this point
     * @param domain
     * @param step the current step in the solution
     * @param optional log that can be used to report convergence status
     */
    virtual bool CheckConvergence(const ablate::domain::Domain& domain, PetscReal time, PetscInt step, const std::shared_ptr<ablate::monitors::logs::Log>& log) = 0;
};

}  // namespace ablate::solver::criteria

#endif  // ABLATELIBRARY_CONVERGENCECRITERIA_HPP

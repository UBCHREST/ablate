#ifndef ABLATELIBRARY_STEADYSTATESTEPPER_HPP
#define ABLATELIBRARY_STEADYSTATESTEPPER_HPP

#include <monitors/logs/log.hpp>
#include "criteria/convergenceCriteria.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::solver {

class SteadyStateStepper : public ablate::solver::TimeStepper {
   private:
    /**
     * The number of steps between criteria checks
     */
    PetscInt checkInterval;

    //! allow for multiple convergence checks
    std::vector<std::shared_ptr<criteria::ConvergenceCriteria>> convergenceCriteria;

    //! optionally log the convergence history
    const std::shared_ptr<ablate::monitors::logs::Log> log = nullptr;

    //! the max number of time steps before giving up
    PetscInt maxSteps = 0;

   public:
    /**
     * constructor for steady state stepper to march the solution to steady state
     * @param domain
     * @param arguments
     * @param initialization
     * @param absoluteTolerances
     * @param relativeTolerances
     * @param verboseSourceCheck
     */
    explicit SteadyStateStepper(std::shared_ptr<ablate::domain::Domain> domain, std::vector<std::shared_ptr<criteria::ConvergenceCriteria>> convergenceCriteria,
                                const std::shared_ptr<ablate::parameters::Parameters> &arguments = {}, std::shared_ptr<ablate::io::Serializer> serializer = {},
                                std::shared_ptr<ablate::domain::Initializer> initialization = {}, std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> absoluteTolerances = {},
                                std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> relativeTolerances = {}, bool verboseSourceCheck = {},
                                std::shared_ptr<ablate::monitors::logs::Log> log = {}, int checkInterval = 0);

    /**
     * clean up any of the local memory
     */
    ~SteadyStateStepper() override;

    /**
     *  Allow Setting up of the criteria
     * @return returns true if it needed to be initialized
     */
    bool Initialize() override;

    /**
     * Solves the system until steady state is achieved
     */
    void Solve() override;
};
}  // namespace ablate::solver

#endif  // ABLATELIBRARY_STEADYSTATESTEPPER_HPP

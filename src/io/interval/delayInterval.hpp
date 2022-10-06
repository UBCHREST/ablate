#ifndef ABLATELIBRARY_DELAYINTERVAL_HPP
#define ABLATELIBRARY_DELAYINTERVAL_HPP

#include <memory>
#include "interval.hpp"

namespace ablate::io::interval {

/**
 * Delays the interval until both the simulation time and step criteria is met.
 */
class DelayInterval : public Interval {
   private:
    //! the base interval to check
    const std::shared_ptr<Interval> interval;

    //! the minimum inclusive simulation time to start the interval checking
    const PetscReal minimumSimulationTime;

    //! the minimum inclusive simulation step to start the interval checking
    const PetscInt minimumSimulationStep;

   public:
    /**
     * Creates a delay interval with the specified base interval
     * @param interval the base interval to check once the requirements are met
     * @param minimumSimulationTime minimum simulation time to start checking the specified interval for true (default is zero)
     * @param minimumStep minimum step time to start checking the specified interval for true (default is zero)
     */
    explicit DelayInterval(std::shared_ptr<Interval> interval, double minimumSimulationTime = 0, int minimumStep = 0);

    /**
     * Function checks the specified interval after the simulation and step criteria are met.
     * @param comm
     * @param steps
     * @param time
     * @return
     */
    bool Check(MPI_Comm comm, PetscInt steps, PetscReal time) override;
};

}  // namespace ablate::io::interval

#endif  // ABLATELIBRARY_WALLTIMEINTERVAL_HPP

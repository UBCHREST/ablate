#ifndef TESTING_RESOURCE_PERRANKASSERT
#define TESTING_RESOURCE_PERRANKASSERT
#include <memory>
#include "assert.hpp"
#include "solver/timeStepper.hpp"

namespace testingResources::asserts {

/**
 * For all asserts/tests that are run during the mpi process run
 */
class PerRankAssert : public Assert {
   public:
    /**
     * Perform any post run/pre process tests using the supplied time stepper
     * @param timeStepper
     */
    virtual void Test(std::shared_pointer<ablate::solver::TimeStepper> timeStepper) = 0;
};

}  // namespace testingResources::asserts

#endif  // TESTING_RESOURCE_PERRANKASSERT
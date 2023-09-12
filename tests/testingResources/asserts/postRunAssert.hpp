#ifndef TESTING_RESOURCE_POSTRUNASSERT
#define TESTING_RESOURCE_POSTRUNASSERT
#include <memory>
#include "assert.hpp"
#include "solver/timeStepper.hpp"

namespace testingResources {
class MpiTestFixture;
}

namespace testingResources::asserts {

/**
 * Runs the tests once after the execution is the mpi process(es) are complete
 */
class PostRunAssert : public Assert {
   public:
    /**
     * Use the mpi test fixture to perform any post execution comparisons/tests
     * @param mpiTestFixture
     */
    virtual void Test(testingResources::MpiTestFixture& mpiTestFixture) = 0;
};

}  // namespace testingResources::asserts

#endif  // TESTING_RESOURCE_POSTRUNASSERT
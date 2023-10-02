#ifndef TESTING_RESOURCE_ASSERT
#define TESTING_RESOURCE_ASSERT

namespace testingResources {

class MpiTestFixture;

namespace asserts {

/**
 * This is an interface that other asserts should utilize
 */
class Assert {
   public:
    virtual ~Assert() = default;
    /**
     * Use the mpi test fixture to perform any post execution comparisons/tests
     * @param mpiTestFixture
     */
    virtual void Test(testingResources::MpiTestFixture& mpiTestFixture) = 0;
};
}  // namespace asserts
}  // namespace testingResources
#endif  // TESTING_RESOURCE_ASSERT
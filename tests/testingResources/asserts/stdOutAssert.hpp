#ifndef TESTING_RESOURCE_STDOUTASSERT
#define TESTING_RESOURCE_STDOUTASSERT
#include "fileAssert.hpp"
#include "postRunAssert.hpp"

namespace testingResources::asserts {

/**
 * Compares the std out stream to an expected log file
 */
class StdOutAssert : public PostRunAssert, private FileAssert {
   private:
    /**
     * The expected log file to compare to
     */
    const std::filesystem::path expectedLogFile;

   public:
    /**
     * Create an assert with the expected log file
     * @param expectedLogFile
     */
    explicit StdOutAssert(std::filesystem::path expectedLogFile);

    /**
     * Compares the generated output file with the expected
     * @param mpiTestFixture
     */
    void Test(testingResources::MpiTestFixture& mpiTestFixture) override;
};

}  // namespace testingResources::asserts

#endif  // TESTING_RESOURCE_POSTRUNASSERT
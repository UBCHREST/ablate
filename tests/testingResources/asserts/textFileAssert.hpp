#ifndef TESTING_RESOURCE_TEXTFILEASSERT
#define TESTING_RESOURCE_TEXTFILEASSERT
#include "assert.hpp"
#include "fileAssert.hpp"

namespace testingResources::asserts {

/**
 * Compares the expected files to those generated
 */
class TextFileAssert : public Assert, private FileAssert {
   private:
    /**
     * The expected log file to compare to
     */
    const std::filesystem::path expectedFile;

    /**
     * The relative path name of the output file in the output directory
     */
    const std::string actualFileName;

   public:
    /**
     * Create an assert with the expected log file
     * @param expectedFile the path to the expected log file (relative to the root of the testing folder)
     * @param actualFileName the file path inside of the output folder
     */
    explicit TextFileAssert(std::filesystem::path expectedFile, std::string actualFileName);

    /**
     * Compares the generated output file with the expected
     * @param mpiTestFixture
     */
    void Test(testingResources::MpiTestFixture& mpiTestFixture) override;
};

}  // namespace testingResources::asserts

#endif  // TESTING_RESOURCE_TEXTFILEASSERT
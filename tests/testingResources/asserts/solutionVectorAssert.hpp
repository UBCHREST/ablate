#ifndef TESTING_RESOURCE_SOLUTIONVECTORASSERT
#define TESTING_RESOURCE_SOLUTIONVECTORASSERT
#include <petsc.h>
#include "assert.hpp"
#include "fileAssert.hpp"
#include "utilities/mathUtilities.hpp"

namespace testingResources::asserts {

/**
 * Compares the expected files to those generated
 */
class SolutionVectorAssert : public Assert, private FileAssert {
   private:
    /**
     * The expected hdf5 file to compare to
     */
    const std::filesystem::path expectedFile;

    /**
     * The relative path name of the output file in the output directory
     */
    const std::string actualFileName;

    /**
     * The norm type to use for the comparison
     */
    ablate::utilities::MathUtilities::Norm normType;

    /**
     * The tolerance for the vector norm comparison
     */
    const PetscReal tolerance;

   public:
    /**
     * Create an assert with the expected log file
     * @param expectedFile the path to the expected hdf5 file (relative to the root of the testing folder)
     * @param actualFileName the file path inside of the output folder
     * @param tolerance the tolerance for the vector norm comparison
     */
    explicit SolutionVectorAssert(std::filesystem::path expectedFile, std::string actualFileName, ablate::utilities::MathUtilities::Norm norm, double tolerance = 1E-12);

    /**
     * Compares the generated output file with the expected
     * @param mpiTestFixture
     */
    void Test(testingResources::MpiTestFixture& mpiTestFixture) override;

   private:
    /**
     * Reads in a solution vector from the specified hdf5 file.
     * @param filePath
     * @param vec
     */
    static void ReadSolutionFromHDFFile(const std::filesystem::path& filePath, Vec* vec);
};

}  // namespace testingResources::asserts

#endif  // TESTING_RESOURCE_TEXTFILEASSERT
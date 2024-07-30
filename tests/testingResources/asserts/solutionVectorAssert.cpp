#include "solutionVectorAssert.hpp"
#include <petscviewerhdf5.h>
#include <utility>
#include "mpiTestFixture.hpp"
#include "utilities/petscUtilities.hpp"

testingResources::asserts::SolutionVectorAssert::SolutionVectorAssert(std::filesystem::path expectedFile, std::string actualFileName, ablate::utilities::MathUtilities::Norm norm, double toleranceIn)
    : expectedFile(std::move(expectedFile)), actualFileName(std::move(actualFileName)), normType(norm), tolerance(toleranceIn == 0.0 ? 1E-12 : toleranceIn) {}

void testingResources::asserts::SolutionVectorAssert::ReadSolutionFromHDFFile(const std::filesystem::path& filePath, Vec* vec) {
    PetscViewer petscViewer = nullptr;

    VecCreate(PETSC_COMM_WORLD, vec) >> ablate::utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)(*vec), "solution") >> ablate::utilities::PetscUtilities::checkError;
    PetscViewerHDF5Open(PETSC_COMM_WORLD, filePath.string().c_str(), FILE_MODE_READ, &petscViewer) >> ablate::utilities::PetscUtilities::checkError;
    PetscViewerHDF5PushGroup(petscViewer, "/fields") >> ablate::utilities::PetscUtilities::checkError;
    PetscViewerHDF5PushTimestepping(petscViewer) >> ablate::utilities::PetscUtilities::checkError;
    PetscViewerHDF5SetTimestep(petscViewer, 0) >> ablate::utilities::PetscUtilities::checkError;
    VecLoad(*vec, petscViewer) >> ablate::utilities::PetscUtilities::checkError;
    PetscViewerDestroy(&petscViewer) >> ablate::utilities::PetscUtilities::checkError;
}

void testingResources::asserts::SolutionVectorAssert::Test(testingResources::MpiTestFixture& mpiTestFixture) {
    // Load the expected solution vector
    std::filesystem::path solutionPath = expectedFile;
    Vec solVec;
    ReadSolutionFromHDFFile(solutionPath, &solVec);

    // Load the actual from the output directory
    std::filesystem::path outputPath = mpiTestFixture.ResultDirectory() / actualFileName;
    Vec outputVec;
    ReadSolutionFromHDFFile(outputPath, &outputVec);

    // Compute the specified norm
    PetscReal norm;
    ablate::utilities::MathUtilities::ComputeNorm(normType, solVec, outputVec, &norm) >> ablate::utilities::PetscUtilities::checkError;
    ASSERT_LT(norm, tolerance);

    // clean up
    VecDestroy(&solVec);
    VecDestroy(&outputVec);
}

#include "registrar.hpp"
REGISTER(testingResources::asserts::Assert, testingResources::asserts::SolutionVectorAssert, "Compares a saved hdf5 solution vector to an expected vector",
         ARG(std::string, "expected", "the path to the expected hdf5 file (relative to the root of the testing folder)"), ARG(std::string, "actual", "the file path inside of the output folder"),
         ENUM(ablate::utilities::MathUtilities::Norm, "type", "norm type ('l1','l1_norm','l2', 'linf', 'l2_norm')"), OPT(double, "tolerance", "the file path inside of the output folder"));
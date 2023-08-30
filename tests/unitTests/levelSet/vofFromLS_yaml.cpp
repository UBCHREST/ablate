#include <petscviewerhdf5.h>
#include <filesystem>
#include "MpiTestFixture.hpp"
#include "MpiTestParamFixture.hpp"
#include "builder.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
#include "petscsys.h"
#include "utilities/petscUtilities.hpp"
#include "yamlParser.hpp"

using namespace ablate;

struct LevelSetParameters_YAML {
    testingResources::MpiTestParameter mpiTestParameter;
    std::filesystem::path yamlFile;
    std::filesystem::path solutionFile;
};

class LevelSetTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<LevelSetParameters_YAML> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

void ReadHDFFile(std::filesystem::path filePath, Vec *vec) {
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

TEST_P(LevelSetTestFixture, ShouldMakeVOFfromLSYaml) {
    StartWithMPI

        {
            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize("Level set to VOF Testing");

            // precompute the resultDirectory directory so we can remove it if it here
            std::filesystem::path resultDirectory = BuildResultDirectory();
            auto testName = TestName();

            // get the input file
            std::filesystem::path yamlFile = GetParam().yamlFile;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"directory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, yamlFile);

            // load a yaml file
            std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(yamlFile);

            // run with the parser
            ablate::Builder::Run(parser);

            // Load the solution vector
            std::filesystem::path solutionPath = GetParam().solutionFile;
            Vec solVec;
            ReadHDFFile(solutionPath, &solVec);

            // Get the output file path. There should be a way to get the the file directly. Need to talk to Matt.
            std::filesystem::path outputPath = ablate::environment::RunEnvironment::Get().GetOutputDirectory() / "domain.hdf5";
            Vec outputVec;
            ReadHDFFile(outputPath, &outputVec);

            // Check if the vectors are equal
            VecAXPY(outputVec, -1.0, solVec);
            PetscReal nrm;
            VecNorm(outputVec, NORM_INFINITY, &nrm);
            ASSERT_LT(nrm, 1e-12);

            VecDestroy(&solVec);
            VecDestroy(&outputVec);
        }
        ablate::environment::RunEnvironment::Finalize();

        exit(0);
    EndWithMPI
}
INSTANTIATE_TEST_SUITE_P(
    LevelSetUnitTests, LevelSetTestFixture,
    testing::Values(
        (LevelSetParameters_YAML){.mpiTestParameter = testingResources::MpiTestParameter("2D_Circle"), .yamlFile = "inputs/levelSet/2D_Circle.yaml", .solutionFile = "outputs/levelSet/2D_Circle.hdf5"},
        (LevelSetParameters_YAML){
            .mpiTestParameter = testingResources::MpiTestParameter("2D_Ellipse"), .yamlFile = "inputs/levelSet/2D_Ellipse.yaml", .solutionFile = "outputs/levelSet/2D_Ellipse.hdf5"},
        (LevelSetParameters_YAML){
            .mpiTestParameter = testingResources::MpiTestParameter("2D_Circle_Tri"), .yamlFile = "inputs/levelSet/2D_Circle_Tri.yaml", .solutionFile = "outputs/levelSet/2D_Circle_Tri.hdf5"}),
    [](const testing::TestParamInfo<LevelSetParameters_YAML> &info) { return info.param.mpiTestParameter.getTestName(); });

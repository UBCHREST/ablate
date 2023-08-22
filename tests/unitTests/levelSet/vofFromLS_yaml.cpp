#include "environment/runEnvironment.hpp"
#include "utilities/petscUtilities.hpp"
#include <filesystem>
#include "MpiTestFixture.hpp"
#include "MpiTestParamFixture.hpp"
#include "builder.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
#include "petscsys.h"
#include "yamlParser.hpp"
#include <petscviewerhdf5.h>
#include <signal.h>
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


//class LevelSetTestFixture : public testingResources::MpiTestParamFixture {};

TEST_P(LevelSetTestFixture, ShouldMakeVOFfromLSYaml) {



    StartWithMPI
//    ASSERT_LT(2, 1);
//raise(SIGSEGV);
        {
printf("Entering the test.\n");
            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize("Level set to VOF Testing");
//            int rank;
//            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            // precompute the resultDirectory directory so we can remove it if it here
            std::filesystem::path resultDirectory = BuildResultDirectory();
//            auto testName = TestName();

//            // get the file
            std::filesystem::path yamlFile = GetParam().yamlFile;

//            std::cout << "resultDirectory: " << resultDirectory << std::endl;
//            std::cout << "inputPath: " << inputPath << std::endl;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"directory", resultDirectory}, {"tagDirectory", "false"}, {"title", yamlFile}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, yamlFile);

            // load a yaml file
            std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(yamlFile);

            // run with the parser
            ablate::Builder::Run(parser);


            // Load the solution vector
            PetscViewer petscViewer = nullptr;
//            std::filesystem::path solutionPath = "solutions/2D_Circle.hdf5";
            std::filesystem::path solutionPath = GetParam().solutionFile;
//            std::cout << "solutionPath: " << solutionPath << std::endl;
            Vec solVec;
            VecCreate(PETSC_COMM_WORLD, &solVec);
            PetscObjectSetName((PetscObject)solVec, "solution");
            PetscViewerHDF5Open(PETSC_COMM_WORLD, solutionPath.string().c_str(), FILE_MODE_READ, &petscViewer) >> ablate::utilities::PetscUtilities::checkError;
            PetscViewerHDF5PushGroup(petscViewer, "/fields");
            PetscViewerHDF5PushTimestepping(petscViewer);
            PetscViewerHDF5SetTimestep(petscViewer, 0);
            VecLoad(solVec, petscViewer);
            PetscViewerDestroy(&petscViewer);



            // Get the output file path. There should be a way to get the the file directly. Need to talk to Matt.
            std::filesystem::path outputPath = ablate::environment::RunEnvironment::Get().GetOutputDirectory() / "domain.hdf5";
            Vec outputVec;
            VecCreate(PETSC_COMM_WORLD, &outputVec);
            PetscObjectSetName((PetscObject)outputVec, "solution");
            PetscViewerHDF5Open(PETSC_COMM_WORLD, outputPath.string().c_str(), FILE_MODE_READ, &petscViewer) >> ablate::utilities::PetscUtilities::checkError;
            PetscViewerHDF5PushGroup(petscViewer, "/fields");
            PetscViewerHDF5PushTimestepping(petscViewer);
            PetscViewerHDF5SetTimestep(petscViewer, 0);
            VecLoad(outputVec, petscViewer);
            PetscViewerDestroy(&petscViewer);

            // Check if the vectors are equal
            VecAXPY(outputVec, -1.0, solVec);
            PetscReal nrm;
            VecNorm(outputVec, NORM_INFINITY, &nrm);
            ASSERT_LT(nrm, 1e-12);



        }
        ablate::environment::RunEnvironment::Finalize();

        exit(0);
    EndWithMPI
}
INSTANTIATE_TEST_SUITE_P(
    LevelSetUnitTests, LevelSetTestFixture,
    testing::Values(
        (LevelSetParameters_YAML){
          .mpiTestParameter = testingResources::MpiTestParameter("2D_Circle"),
          .yamlFile = "inputs/levelSet/2D_Circle.yaml",
          .solutionFile = "outputs/levelSet/2D_Circle.hdf5"}));



//INSTANTIATE_TEST_SUITE_P(
//    MeshTests, RBFTestFixture_RBFValues,
//    testing::Values(
//        (RBFParameters_RBFValues){.mpiTestParameter = testingResources::MpiTestParameter("inputs/2D_Circle.yaml"),
//                                  .solutionFile = "outputs/2d_Circle.hdf5"})
//    [](const testing::TestParamInfo<RBFParameters_RBFValues> &info) { return info.param.mpiTestParameter.getTestName(); });

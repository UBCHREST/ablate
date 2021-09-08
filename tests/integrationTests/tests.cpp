static char help[] = "Integration Level Testing";

#include <filesystem>
#include "MpiTestFixture.hpp"
#include "MpiTestParamFixture.hpp"
#include "builder.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
#include "parser/yamlParser.hpp"
#include "petscsys.h"

/**
 * Note: the test name is assumed to be the relative path to the yaml file
 */
class IntegrationTestsSpecifier : public testingResources::MpiTestParamFixture {};

TEST_P(IntegrationTestsSpecifier, ShouldRun) {
    StartWithMPI
        // initialize petsc and mpi
        if (!PETSC_USE_LOG) {
            FAIL() << "Integration testing requires PETSC_LOG";
        }
        PetscOptionsSetValue(NULL, "-objects_dump", NULL) >> testErrorChecker;
        PetscInitialize(argc, argv, NULL, help) >> testErrorChecker;
        {
            int rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            // precompute the resultDirectory directory so we can remove it if it here
            auto testName = GetParam().getTestName();
            std::filesystem::path resultDirectory = std::filesystem::current_path() / testName;
            if (rank == 0) {
                std::filesystem::remove_all(resultDirectory);
            }
            MPI_Barrier(PETSC_COMM_WORLD);

            // get the file
            std::filesystem::path inputPath = GetParam().testName;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"outputDirectory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

            // load a yaml file
            std::shared_ptr<ablate::parser::Factory> parser = std::make_shared<ablate::parser::YamlParser>(inputPath);

            // run with the parser
            ablate::Builder::Run(parser);

            // print all files in the directory so that they are compared with expected
            if (rank == 0) {
                std::vector<std::string> resultFileInfo;
                for (const auto& entry : fs::directory_iterator(ablate::environment::RunEnvironment::Get().GetOutputDirectory())) {
                    resultFileInfo.push_back(entry.path().filename());
                }
                // sort the names so that the output order is defined
                std::sort(resultFileInfo.begin(), resultFileInfo.end());
                std::cout << "ResultFiles:" << std::endl;
                for (const auto& fileInfo : resultFileInfo) {
                    std::cout << fileInfo << std::endl;
                }
            }
        }
        PetscFinalize() >> testErrorChecker;
        exit(0);
    EndWithMPI
}

struct IntegrationRestartTestsParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::map<std::string, std::string> restartOverrides;
};

class IntegrationRestartTestsSpecifier : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<IntegrationRestartTestsParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(IntegrationRestartTestsSpecifier, ShouldRunAndRestart) {
    // First Run and setup
    StartWithMPI
        // initialize petsc and mpi
        PetscOptionsSetValue(NULL, "-objects_dump", NULL) >> testErrorChecker;
        PetscInitialize(argc, argv, NULL, help) >> testErrorChecker;
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        // precompute the resultDirectory directory so we can remove it if it is here
        auto testName = GetParam().mpiTestParameter.getTestName();
        std::filesystem::path resultDirectory = std::filesystem::current_path() / testName;
        if (rank == 0) {
            std::filesystem::remove_all(resultDirectory);
        }
        MPI_Barrier(PETSC_COMM_WORLD);

        // Perform the initial run
        {
            // get the file
            std::filesystem::path inputPath = GetParam().mpiTestParameter.testName;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"outputDirectory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

            // load a yaml file
            std::shared_ptr<ablate::parser::Factory> parser = std::make_shared<ablate::parser::YamlParser>(inputPath);

            // run with the parser
            ablate::Builder::Run(parser);
        }

        // create a new result directory
        auto restartResultDirectory = std::filesystem::current_path() / (testName + "_resume");
        if (rank == 0) {
            std::filesystem::remove_all(restartResultDirectory);
        }
        MPI_Barrier(PETSC_COMM_WORLD);

        // Restart the simulation
        {
            // precompute the restart directory
            std::filesystem::path restartPath = resultDirectory / "restart.rst";
            ASSERT_TRUE(std::filesystem::exists(restartPath));

            // get the input path from the parser
            std::filesystem::path inputPath = GetParam().mpiTestParameter.testName;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"outputDirectory", restartResultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

            // override some parameters
            auto overrideMap = GetParam().restartOverrides;
            overrideMap["restart::restartFile"] = restartPath;
            auto overrideParams = std::make_shared<ablate::parameters::MapParameters>(overrideMap);

            // load a yaml file
            std::shared_ptr<ablate::parser::Factory> parser = std::make_shared<ablate::parser::YamlParser>(inputPath, false, overrideParams);

            // run with the parser
            ablate::Builder::Run(parser);
        }

        // print all files in the directory so that they are compared with expected
        if (rank == 0) {
            std::vector<std::string> resultFileInfo;
            for (const auto& entry : fs::directory_iterator(resultDirectory)) {
                resultFileInfo.push_back(entry.path().filename());
            }
            // sort the names so that the output order is defined
            std::sort(resultFileInfo.begin(), resultFileInfo.end());
            std::cout << "ResultFiles:" << std::endl;
            for (const auto& fileInfo : resultFileInfo) {
                std::cout << fileInfo << std::endl;
            }

            resultFileInfo.clear();
            for (const auto& entry : fs::directory_iterator(restartResultDirectory)) {
                resultFileInfo.push_back(entry.path().filename());
            }
            // sort the names so that the output order is defined
            std::sort(resultFileInfo.begin(), resultFileInfo.end());
            std::cout << "RestartResultFiles:" << std::endl;
            for (const auto& fileInfo : resultFileInfo) {
                std::cout << fileInfo << std::endl;
            }
        }

        PetscFinalize() >> testErrorChecker;
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    Tests, IntegrationTestsSpecifier,
    testing::Values((MpiTestParameter){.testName = "inputs/compressibleCouetteFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleCouetteFlow.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/incompressibleFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/incompressibleFlow.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/tracerParticles2DHDF5Monitor.yaml", .nproc = 2, .expectedOutputFile = "outputs/tracerParticles2DHDF5Monitor.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/tracerParticles3D.yaml", .nproc = 1, .expectedOutputFile = "outputs/tracerParticles3D.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/compressibleFlowVortex.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleFlowVortex.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/customCouetteCompressibleFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/customCouetteCompressibleFlow.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/simpleReactingFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/simpleReactingFlow.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/ignitionDelayGriMech.yaml", .nproc = 1, .expectedOutputFile = "outputs/ignitionDelayGriMech.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/dmViewFromOptions.yaml", .nproc = 1, .expectedOutputFile = "outputs/dmViewFromOptions.txt", .arguments = ""}),
    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(Tests, IntegrationRestartTestsSpecifier,
                         testing::Values(
                             (IntegrationRestartTestsParameters){
                                 .mpiTestParameter = {.testName = "inputs/incompressibleFlowRestart.yaml", .nproc = 1, .expectedOutputFile = "outputs/incompressibleFlowRestart.txt", .arguments = ""},
                                 .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
                             (IntegrationRestartTestsParameters){
                                 .mpiTestParameter = {.testName = "inputs/incompressibleFlowRestart.yaml", .nproc = 2, .expectedOutputFile = "outputs/incompressibleFlowRestart.txt", .arguments = ""},
                                 .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}}),
                         [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) { return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc); });
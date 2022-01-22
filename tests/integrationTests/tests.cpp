static char help[] = "Integration Level Testing";

#include <filesystem>
#include <utilities/fileUtility.hpp>
#include "MpiTestFixture.hpp"
#include "MpiTestParamFixture.hpp"
#include "builder.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
#include "petscsys.h"
#include "yamlParser.hpp"

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
        PetscOptionsSetValue(NULL, "-checkstack", "true") >> testErrorChecker;
        PetscInitialize(argc, argv, NULL, help) >> testErrorChecker;
        {
            int rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            // precompute the resultDirectory directory so we can remove it if it here
            std::filesystem::path resultDirectory = BuildResultDirectory();
            auto testName = TestName();

            // get the file
            std::filesystem::path inputPath = GetParam().testName;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"directory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

            {
                // load a yaml file
                ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {inputPath.parent_path()});
                std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(inputPath, fileLocator.GetLocateFileFunction());

                // run with the parser
                ablate::Builder::Run(parser);
            }
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
        PetscOptionsSetValue(NULL, "-checkstack", "true") >> testErrorChecker;
        PetscInitialize(argc, argv, NULL, help) >> testErrorChecker;
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        // precompute the resultDirectory directory so we can remove it if it is here
        std::filesystem::path resultDirectory = BuildResultDirectory();
        auto testName = TestName();

        // Perform the initial run
        {
            // get the file
            std::filesystem::path inputPath = GetParam().mpiTestParameter.testName;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"directory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

            // load a yaml file
            ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {inputPath.parent_path()});
            std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(inputPath, fileLocator.GetLocateFileFunction());

            // run with the parser
            ablate::Builder::Run(parser);
        }

        // Restart the simulation
        {
            // get the input path from the parser
            std::filesystem::path inputPath = GetParam().mpiTestParameter.testName;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"directory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

            // override some parameters
            auto overrideMap = GetParam().restartOverrides;

            // load a yaml file
            ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {inputPath.parent_path()});
            std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(inputPath, fileLocator.GetLocateFileFunction(), overrideMap);

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
        }

        PetscFinalize() >> testErrorChecker;
        exit(0);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    Tests, IntegrationTestsSpecifier,
    testing::Values(
        (MpiTestParameter){.testName = "inputs/compressibleCouetteFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleCouetteFlow.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/incompressibleFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/incompressibleFlow.txt", .arguments = ""},
        (MpiTestParameter){
            .testName = "inputs/tracerParticles2DHDF5Monitor.yaml",
            .nproc = 2,
            .expectedOutputFile = "outputs/tracerParticles2DHDF5Monitor.txt",
            .arguments = "",
            .expectedFiles{{"outputs/tracerParticles2DHDF5Monitor/flowTracerParticles.xmf", "flowTracerParticles.xmf"}, {"outputs/tracerParticles2DHDF5Monitor/domain.xmf", "domain.xmf"}}},
        (MpiTestParameter){.testName = "inputs/tracerParticles3D.yaml", .nproc = 1, .expectedOutputFile = "outputs/tracerParticles3D.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/shockTubeRieman.yaml", .nproc = 1, .expectedOutputFile = "outputs/shockTubeRieman.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/compressibleFlowVortex.yaml",
                           .nproc = 1,
                           .expectedOutputFile = "outputs/compressibleFlowVortex.txt",
                           .arguments = "",
                           .expectedFiles{{"outputs/compressibleFlowVortex/domain.xmf", "domain.xmf"}}},
        (MpiTestParameter){.testName = "inputs/customCouetteCompressibleFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/customCouetteCompressibleFlow.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/simpleReactingFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/simpleReactingFlow.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/ignitionDelayGriMech.yaml", .nproc = 1, .arguments = "", .expectedFiles{{"outputs/ignitionDelayGriMech.PeakYi.txt", "ignitionDelayPeakYi.txt"}}},
        (MpiTestParameter){
            .testName = "inputs/ignitionDelay2S_CH4_CM2.yaml", .nproc = 1, .arguments = "", .expectedFiles{{"outputs/ignitionDelay2S_CH4_CM2.Temperature.txt", "ignitionDelayTemperature.txt"}}},
        (MpiTestParameter){.testName = "inputs/dmViewFromOptions.yaml", .nproc = 1, .expectedOutputFile = "outputs/dmViewFromOptions.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/extraVariableTransport.yaml", .nproc = 1, .expectedOutputFile = "outputs/extraVariableTransport.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/subDomainFVM.yaml",
                           .nproc = 1,
                           .expectedOutputFile = "outputs/subDomainFVM/subDomainFVM.txt",
                           .arguments = "",
                           .expectedFiles{{"outputs/subDomainFVM/fluidField.xmf", "fluidField.xmf"}}},
        (MpiTestParameter){.testName = "inputs/shockTubeSODLodiBoundary.yaml", .nproc = 1, .expectedOutputFile = "outputs/shockTubeSODLodiBoundary.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/steadyCompressibleFlowLodiTest.yaml", .nproc = 2, .expectedOutputFile = "outputs/steadyCompressibleFlowLodiTest.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/compressibleFlowVortexLodi.yaml", .nproc = 2, .expectedOutputFile = "outputs/compressibleFlowVortexLodi.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/twoGasAdvectingDiscontinuity.yaml", .nproc = 1, .expectedOutputFile = "outputs/twoGasAdvectingDiscontinuity.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/compressibleFlowPgsLodi.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleFlowPgsLodi.txt", .arguments = ""}),
    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(
    Tests, IntegrationRestartTestsSpecifier,
    testing::Values(
        (IntegrationRestartTestsParameters){
            .mpiTestParameter = {.testName = "inputs/incompressibleFlowRestart.yaml", .nproc = 1, .expectedOutputFile = "outputs/incompressibleFlowRestart.txt", .arguments = ""},
            .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
        (IntegrationRestartTestsParameters){
            .mpiTestParameter = {.testName = "inputs/incompressibleFlowRestart.yaml", .nproc = 2, .expectedOutputFile = "outputs/incompressibleFlowRestart.txt", .arguments = ""},
            .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
        (IntegrationRestartTestsParameters){
            .mpiTestParameter = {.testName = "inputs/tracerParticles2DRestart.yaml", .nproc = 1, .expectedOutputFile = "outputs/tracerParticles2DRestart.txt", .arguments = ""},
            .restartOverrides = {{"timestepper::arguments::ts_max_steps", "10"}}},
        (IntegrationRestartTestsParameters){
            .mpiTestParameter = {.testName = "inputs/incompressibleFlowIntervalRestart.yaml", .nproc = 1, .expectedOutputFile = "outputs/incompressibleFlowIntervalRestart.txt", .arguments = ""},
            .restartOverrides = {{"timestepper::arguments::ts_max_steps", "10"}}}),
    [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) { return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc); });
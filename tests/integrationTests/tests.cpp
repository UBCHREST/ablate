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

INSTANTIATE_TEST_SUITE_P(
    Tests, IntegrationTestsSpecifier,
    testing::Values((MpiTestParameter){.testName = "inputs/compressibleCouetteFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleCouetteFlow.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/incompressibleFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/incompressibleFlow.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/tracerParticles2DHDF5Monitor.yaml", .nproc = 2, .expectedOutputFile = "outputs/tracerParticles2DHDF5Monitor.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/tracerParticles3D.yaml", .nproc = 1, .expectedOutputFile = "outputs/tracerParticles3D.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/compressibleFlowVortex.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleFlowVortex.txt", .arguments = ""}),
    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

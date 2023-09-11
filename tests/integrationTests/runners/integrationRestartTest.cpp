#include "integrationRestartTest.hpp"

#include <utility>
#include "environment/runEnvironment.hpp"
#include "utilities/petscUtilities.hpp"

IntegrationRestartTestFixture::IntegrationRestartTestFixture(std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter, std::string restartInputFile,
                                                             std::shared_ptr<ablate::parameters::Parameters> restartOverrides): IntegrationTestFixture(std::move(mpiTestParameter)), restartInputFile(std::move(restartInputFile)), restartOverrides(std::move(restartOverrides)) {}

IntegrationRestartTest::IntegrationRestartTest(std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter, std::string restartInputFile,
                                               std::shared_ptr<ablate::parameters::Parameters> restartOverrides)
    : IntegrationRestartTestFixture(std::move(mpiTestParameter), std::move(restartInputFile), std::move(restartOverrides)) {}

void IntegrationRestartTest::RegisterTest(const std::filesystem::path& inputPath) {
    // make a raw pointer copy of the integration test
    auto integrationTestCopy = new IntegrationRestartTest(mpiTestParameter, restartInputFile, restartOverrides);
    integrationTestCopy->inputFilePath = inputPath;

    // check to see if it contains a test target
    testing::RegisterTest(integrationTestCopy->GetTestSuiteName(std::filesystem::path("inputs"), "IntegrationRestart").c_str(),
                          integrationTestCopy->GetTestName().c_str(),
                          nullptr,
                          nullptr,
                          absolute(inputPath).c_str(),
                          1,
                          // Important to use the fixture type as the return type here.
                          [=]() -> IntegrationRestartTestFixture* { return integrationTestCopy; });
}

void IntegrationRestartTest::TestBody() {
        // First Run and setup
        StartWithMPI
            // initialize petsc and mpi
            PetscOptionsSetValue(NULL, "-objects_dump", NULL) >> testErrorChecker;
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize("Integration Level Testing");

            int rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            // precompute the resultDirectory directory so we can remove it if it is here
            std::filesystem::path resultDirectory = BuildResultDirectory();
            auto testName = TestName();

            // Perform the initial run
            {
                // get the file
                std::filesystem::path inputPath = inputFilePath;

                // Setup the run environment
                ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"directory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
                ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

                // load a yaml file
                std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(inputPath);

                // run with the parser
                ablate::Builder::Run(parser);
            }

            // Restart the simulation
            {
                // get the input path from the parser
                std::filesystem::path inputPath = restartInputFile.empty() ? inputFilePath : std::filesystem::path(restartInputFile);

                // Setup the run environment
                ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"directory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
                ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

                // override some parameters
                auto overrideMap = restartOverrides ? restartOverrides->ToMap<std::string>() : std::map<std::string, std::string>{};

                // load a yaml file
                std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(inputPath, overrideMap);

                // run with the parser
                ablate::Builder::Run(parser);
            }

            // print all files in the directory so that they are compared with expected
            if (rank == 0) {
                std::vector<std::string> resultFileInfo;
                for (const auto& entry : fs::recursive_directory_iterator(resultDirectory)) {
                    resultFileInfo.push_back(entry.path().filename());
                }
                // sort the names so that the output order is defined
                std::sort(resultFileInfo.begin(), resultFileInfo.end());
                std::cout << "ResultFiles:" << std::endl;
                for (const auto& fileInfo : resultFileInfo) {
                    std::cout << fileInfo << std::endl;
                }
            }

            ablate::environment::RunEnvironment::Finalize();
            exit(0);
        EndWithMPI
}

#include "registrar.hpp"
REGISTER(IntegrationTestFixture, IntegrationRestartTest,
         "Runs the basic ABLATE integration test",
         ARG(testingResources::MpiTestParameter, "testParameters", "the basic mpi test parameters"),
         OPT(std::string, "restartInputFile", "optional other input file to use upon restart"),
         OPT(ablate::parameters::Parameters, "restartOverrides", "overrides for the input file upon restart")
         );

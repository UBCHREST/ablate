#include "integrationTest.hpp"

#include <utility>
#include "environment/runEnvironment.hpp"
#include "utilities/petscUtilities.hpp"

IntegrationTestFixture::IntegrationTestFixture(std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter) : mpiTestParameter(std::move(mpiTestParameter)) {}

IntegrationTest::IntegrationTest(std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter) : IntegrationTestFixture(std::move(mpiTestParameter)) {}

void IntegrationTest::RegisterTest(const std::filesystem::path& inputPath) {
    inputFilePath = inputPath;

    // get a copy of this pointer so that this lambda can prevent deletion
    auto testPointer = shared_from_this();

    // check to see if it contains a test target
    testing::RegisterTest(GetTestSuiteName(std::filesystem::path("inputs")).c_str(),
                          GetTestName().c_str(),
                          nullptr,
                          nullptr,
                          absolute(inputPath).c_str(),
                          1,
                          // Important to use the fixture type as the return type here.
                          [testPointer]() -> IntegrationTestFixture* {
                              auto newTestPointer = new IntegrationTest(testPointer->mpiTestParameter);
                              newTestPointer->inputFilePath = testPointer->inputFilePath;
                              return newTestPointer;
                          });
}

void IntegrationTest::TestBody() {
    StartWithMPI
        // initialize petsc and mpi
        if (!PETSC_USE_LOG) {
            FAIL() << "Integration testing requires PETSC_LOG";
        }
        PetscOptionsSetValue(nullptr, "-objects_dump", nullptr) >> testErrorChecker;
        ablate::environment::RunEnvironment::Initialize(argc, argv);
        ablate::utilities::PetscUtilities::Initialize("Integration Level Testing");
        {
            int rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

            // precompute the resultDirectory directory so we can remove it if it here
            std::filesystem::path resultDirectory = BuildResultDirectory();
            auto testName = TestName();

            // get the file
            std::filesystem::path inputPath = inputFilePath;

            // Setup the run environment
            ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"directory", resultDirectory}, {"tagDirectory", "false"}, {"title", testName}});
            ablate::environment::RunEnvironment::Setup(runEnvironmentParameters, inputPath);

            {
                // load a yaml file
                std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(inputPath);

                // get the time stepper
                auto timeStepper = ablate::Builder::Build(parser);

                // run with the parser
                timeStepper->Solve();
            }
            // print all files in the directory so that they are compared with expected
            if (rank == 0) {
                std::vector<std::string> resultFileInfo;
                for (const auto& entry : fs::recursive_directory_iterator(ablate::environment::RunEnvironment::Get().GetOutputDirectory())) {
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
        ablate::environment::RunEnvironment::Finalize();
        exit(0);
    EndWithMPI
}

#include "registrar.hpp"
REGISTER_DEFAULT_PASS_THROUGH(IntegrationTestFixture, IntegrationTest, "Runs the basic ABLATE integration test", testingResources::MpiTestParameter);

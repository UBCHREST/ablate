#include "runners.hpp"
#include "environment/runEnvironment.hpp"
#include "utilities/petscUtilities.hpp"

TEST_P(RegressionTestsSpecifier, ShouldRun) {
    StartWithMPI
        // initialize petsc and mpi
        if (!PETSC_USE_LOG) {
            FAIL() << "Regression testing requires PETSC_LOG";
        }
        PetscOptionsSetValue(NULL, "-objects_dump", NULL) >> testErrorChecker;
        ablate::environment::RunEnvironment::Initialize(argc, argv);
        ablate::utilities::PetscUtilities::Initialize("Regression Level Testing");
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
                std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(inputPath);

                // run with the parser
                ablate::Builder::Run(parser);
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
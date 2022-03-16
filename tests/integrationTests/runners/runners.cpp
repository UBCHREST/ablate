#include "runners.hpp"

TEST_P(IntegrationTestsSpecifier, ShouldRun) {
    StartWithMPI
        // initialize petsc and mpi
        if (!PETSC_USE_LOG) {
            FAIL() << "Integration testing requires PETSC_LOG";
        }
        PetscOptionsSetValue(NULL, "-objects_dump", NULL) >> testErrorChecker;
        PetscOptionsSetValue(NULL, "-checkstack", "true") >> testErrorChecker;
        PetscInitialize(argc, argv, NULL, "Integration Level Testing") >> testErrorChecker;
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
        PetscFinalize() >> testErrorChecker;
        exit(0);
    EndWithMPI
}

TEST_P(IntegrationRestartTestsSpecifier, ShouldRunAndRestart) {
    // First Run and setup
    StartWithMPI
        // initialize petsc and mpi
        PetscOptionsSetValue(NULL, "-objects_dump", NULL) >> testErrorChecker;
        PetscOptionsSetValue(NULL, "-checkstack", "true") >> testErrorChecker;
        PetscInitialize(argc, argv, NULL, "Integration Level Testing") >> testErrorChecker;
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

        PetscFinalize() >> testErrorChecker;
        exit(0);
    EndWithMPI
}
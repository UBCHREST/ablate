#include "mpiTestFixture.hpp"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include "asserts/postRunAssert.hpp"

int* testingResources::MpiTestFixture::argc;
char*** testingResources::MpiTestFixture::argv;
const std::string testingResources::MpiTestFixture::InTestRunFlag = "--runMpiTestDirectly=true";
const std::string testingResources::MpiTestFixture::Test_Mpi_Command_Name = "TEST_MPI_COMMAND";
const std::string testingResources::MpiTestFixture::Keep_Output_File = "--keepOutputFile=true";

bool testingResources::MpiTestFixture::inMpiTestRun;
bool testingResources::MpiTestFixture::keepOutputFile;

#if defined(COMPILE_MPI_COMMAND)
#define STR1(x) #x
#define STR(x) STR1(x)
std::string testingResources::MpiTestFixture::mpiCommand = STR(COMPILE_MPI_COMMAND);
#else
std::string testingResources::MpiTestFixture::mpiCommand = "mpirun";
#endif

std::string testingResources::MpiTestFixture::ParseCommandLineArgument(int* argcIn, char*** argvIn, const std::string flag) {
    int commandLineArgumentLocation = -1;
    std::string argument;
    for (auto i = 0; i < *argcIn; i++) {
        if (strncmp(flag.c_str(), (*argvIn)[i], flag.length()) == 0) {
            commandLineArgumentLocation = i;

            argument = std::string((*argvIn)[i]);
            argument = argument.substr(argument.find("=") + 1);
        }
    }

    if (commandLineArgumentLocation >= 0) {
        *argcIn = (*argcIn) - 1;
        for (auto i = commandLineArgumentLocation; i < *argcIn; i++) {
            (*argvIn)[i] = (*argvIn)[i + 1];
        }
    }

    return argument;
}

bool testingResources::MpiTestFixture::InitializeTestingEnvironment(int* argcIn, char*** argvIn) {
    MpiTestFixture::argc = argcIn;
    MpiTestFixture::argv = argvIn;

    MpiTestFixture::inMpiTestRun = !MpiTestFixture::ParseCommandLineArgument(MpiTestFixture::argc, MpiTestFixture::argv, MpiTestFixture::InTestRunFlag).empty();
    MpiTestFixture::keepOutputFile = !MpiTestFixture::ParseCommandLineArgument(MpiTestFixture::argc, MpiTestFixture::argv, MpiTestFixture::Keep_Output_File).empty();

    char* commandName = std::getenv(MpiTestFixture::Test_Mpi_Command_Name.c_str());
    if (commandName != nullptr) {
        MpiTestFixture::mpiCommand = std::string(commandName);
    }

    return inMpiTestRun;
}

void testingResources::MpiTestFixture::SetUp() {}

void testingResources::MpiTestFixture::TearDown() {
    if (!inMpiTestRun) {
        if (!MpiTestFixture::keepOutputFile) {
            fs::remove_all(OutputFile());
        } else {
            std::cout << "Generated output file: " << OutputFile() << std::endl;
        }
    }
}

void testingResources::MpiTestFixture::RunWithMPI() const {
    // build the mpi command
    std::stringstream mpiCommandBuild;
    // Build the asan or other environment flags if needed
    if (!mpiTestParameter.environment.empty()) {
        mpiCommandBuild << mpiTestParameter.environment << " ";
    }
    mpiCommandBuild << MpiTestFixture::mpiCommand << " ";
    mpiCommandBuild << "-n " << mpiTestParameter.nproc << " ";
    mpiCommandBuild << "\"" << ExecutablePath() << "\" ";
    mpiCommandBuild << InTestRunFlag << " ";
    mpiCommandBuild << "--gtest_filter=" << TestName() << " ";
    mpiCommandBuild << mpiTestParameter.arguments << " ";
    mpiCommandBuild << " > " << OutputFile() << " 2>&1";

    auto exitCode = std::system(mpiCommandBuild.str().c_str());
    if (exitCode != 0) {
        std::ifstream outputStream(OutputFile());
        std::string output((std::istreambuf_iterator<char>(outputStream)), std::istreambuf_iterator<char>());
        FAIL() << output;
    }
}

void testingResources::MpiTestFixture::CheckAsserts() {
    // Run through each post mpi check
    for (auto& assert : mpiTestParameter.asserts) {
        if (auto postRunAssert = std::dynamic_pointer_cast<asserts::PostRunAssert>(assert)) {
            if (postRunAssert) {
                postRunAssert->Test(*this);
            }
        }
    }
}

std::filesystem::path testingResources::MpiTestFixture::BuildResultDirectory() const {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    std::filesystem::path resultDirectory = ResultDirectory();
    if (rank == 0) {
        std::filesystem::remove_all(resultDirectory);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
    return resultDirectory;
}

#include "registrar.hpp"
REGISTER_DEFAULT(testingResources::MpiTestParameter, testingResources::MpiTestParameter, "Specifies how an MPI Test will run", ARG(std::string, "name", "the test name"),
                 OPT(int, "ranks", "the number of MPI ranks (default is 1)"), OPT(std::string, "arguments", "Optional program arguments"),
                 OPT(std::string, "environment", "Add options for ASAN flags"),
                 OPT(testingResources::asserts::Assert, "assert", "a single assert object that can be used to determine if the case passed or failed"),
                 OPT(std::vector<testingResources::asserts::Assert>, "asserts", "list of assert objects that can be used to determine if the case passed or failed"));
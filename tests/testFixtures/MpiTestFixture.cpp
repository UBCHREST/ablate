#include "MpiTestFixture.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>

int* MpiTestFixture::argc;
char*** MpiTestFixture::argv;
const std::string MpiTestFixture::InTestRunFlag = "--inmpitestrun=true";
const std::string MpiTestFixture::Test_Mpi_Command_Name = "TEST_MPI_COMMAND";

bool MpiTestFixture::inMpiTestRun;
std::string MpiTestFixture::mpiCommand = "mpirun";

std::string MpiTestFixture::ParseCommandLineArgument(int* argc, char*** argv, const std::string flag) {
    int commandLineArgumentLocation = -1;
    std::string argument;
    for (auto i = 0; i < *argc; i++) {
        if (strncmp(flag.c_str(), (*argv)[i], flag.length()) == 0) {
            commandLineArgumentLocation = i;

            argument = std::string((*argv)[i]);
            argument = argument.substr(argument.find("=") + 1);
        }
    }

    if (commandLineArgumentLocation >= 0) {
        *argc = (*argc) - 1;
        for (auto i = commandLineArgumentLocation; i < *argc; i++) {
            (*argv)[i] = (*argv)[i + 1];
        }
    }

    return argument;
}

bool MpiTestFixture::InitializeTestingEnvironment(int* argc, char*** argv) {
    MpiTestFixture::argc = argc;
    MpiTestFixture::argv = argv;

    MpiTestFixture::inMpiTestRun = !MpiTestFixture::ParseCommandLineArgument(MpiTestFixture::argc, MpiTestFixture::argv, MpiTestFixture::InTestRunFlag).empty();

    char* commandName = std::getenv(MpiTestFixture::Test_Mpi_Command_Name.c_str());
    if (commandName != NULL) {
        MpiTestFixture::mpiCommand = std::string(commandName);
    }

    return inMpiTestRun;
}

void MpiTestFixture::SetUp() {}

void MpiTestFixture::TearDown() {
    if (!inMpiTestRun) {
        fs::remove_all(OutputFile());
    }
}

void MpiTestFixture::RunWithMPI() const {
    // build the mpi command
    std::stringstream mpiCommand;
    mpiCommand << MpiTestFixture::mpiCommand << " ";
    mpiCommand << "-n " << mpiTestParameter.nproc << " ";
    mpiCommand << "\"" << ExecutablePath() << "\" ";
    mpiCommand << InTestRunFlag << " ";
    mpiCommand << "--gtest_filter=" << TestName() << " ";
    mpiCommand << mpiTestParameter.arguments << " ";
    mpiCommand << " > " << OutputFile();

    auto exitCode = std::system(mpiCommand.str().c_str());
    if(exitCode != 0){
        std::ifstream outputStream(OutputFile());
        std::string output((std::istreambuf_iterator<char>(outputStream)), std::istreambuf_iterator<char>());
        FAIL() << output;
    }
}

void MpiTestFixture::CompareOutputFiles() {
    if(mpiTestParameter.expectedOutputFile.empty()){
        return;
    }
    // load the actual output
    std::ifstream actualStream(OutputFile());
    std::string actual((std::istreambuf_iterator<char>(actualStream)), std::istreambuf_iterator<char>());

    // read in the expected
    std::ifstream expectedStream(mpiTestParameter.expectedOutputFile);
    std::string expected((std::istreambuf_iterator<char>(expectedStream)), std::istreambuf_iterator<char>());

    ASSERT_TRUE(actual.length() > 0) << "Actual output is expected not to be empty";
    ASSERT_EQ(actual, expected);
}

std::ostream& operator<<(std::ostream& os, const MpiTestParameter& params) {
    return os << (params.expectedOutputFile.empty() ? params.arguments : params.expectedOutputFile);
}
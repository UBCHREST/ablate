#include "MpiTestFixture.hpp"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>

int* testingResources::MpiTestFixture::argc;
char*** testingResources::MpiTestFixture::argv;
const std::string testingResources::MpiTestFixture::InTestRunFlag = "--runMpiTestDirectly=true";
const std::string testingResources::MpiTestFixture::Test_Mpi_Command_Name = "TEST_MPI_COMMAND";
const std::string testingResources::MpiTestFixture::Keep_Output_File = "--keepOutputFile=true";
const std::string expectedResultDelimiter = std::string("<expects>");

bool testingResources::MpiTestFixture::inMpiTestRun;
bool testingResources::MpiTestFixture::keepOutputFile;

std::string testingResources::MpiTestFixture::mpiCommand = "mpirun";

std::string testingResources::MpiTestFixture::ParseCommandLineArgument(int* argc, char*** argv, const std::string flag) {
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

bool testingResources::MpiTestFixture::InitializeTestingEnvironment(int* argc, char*** argv) {
    MpiTestFixture::argc = argc;
    MpiTestFixture::argv = argv;

    MpiTestFixture::inMpiTestRun = !MpiTestFixture::ParseCommandLineArgument(MpiTestFixture::argc, MpiTestFixture::argv, MpiTestFixture::InTestRunFlag).empty();
    MpiTestFixture::keepOutputFile = !MpiTestFixture::ParseCommandLineArgument(MpiTestFixture::argc, MpiTestFixture::argv, MpiTestFixture::Keep_Output_File).empty();

    char* commandName = std::getenv(MpiTestFixture::Test_Mpi_Command_Name.c_str());
    if (commandName != NULL) {
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
    std::stringstream mpiCommand;
    mpiCommand << MpiTestFixture::mpiCommand << " ";
    mpiCommand << "-n " << mpiTestParameter.nproc << " ";
    mpiCommand << "\"" << ExecutablePath() << "\" ";
    mpiCommand << InTestRunFlag << " ";
    mpiCommand << "--gtest_filter=" << TestName() << " ";
    mpiCommand << mpiTestParameter.arguments << " ";
    mpiCommand << " > " << OutputFile();

    auto exitCode = std::system(mpiCommand.str().c_str());
    if (exitCode != 0) {
        std::ifstream outputStream(OutputFile());
        std::string output((std::istreambuf_iterator<char>(outputStream)), std::istreambuf_iterator<char>());
        FAIL() << output;
    }
}

void testingResources::MpiTestFixture::CompareOutputFiles() {
    if (mpiTestParameter.expectedOutputFile.empty()) {
        return;
    }
    // load in the actual and expected results files
    if (!std::filesystem::exists(mpiTestParameter.expectedOutputFile)) {
        FAIL() << "The expected output file " << mpiTestParameter.expectedOutputFile << " cannot be found";
    }

    std::ifstream expectedStream(mpiTestParameter.expectedOutputFile);
    std::ifstream actualStream(OutputFile());

    // march over each line
    std::string expectedLine;
    std::string actualLine;
    int lineNumber = 1;
    while (std::getline(expectedStream, expectedLine)) {
        if (!std::getline(actualStream, actualLine)) {
            FAIL() << "The actual output file is missing lines";
        }

        // check to see if this lines includes any expected values
        auto expectedResultDelimiterPosition = expectedLine.find(expectedResultDelimiter);
        if (expectedResultDelimiterPosition == std::string::npos) {
            // do a direct match
            ASSERT_EQ(expectedLine, actualLine);
        } else {
            std::string regexLine = expectedLine.substr(0, expectedResultDelimiterPosition);
            std::string valuesLine = expectedLine.substr(expectedResultDelimiterPosition + expectedResultDelimiter.size());

            // get the matches
            std::smatch matches;
            std::regex_search(actualLine, matches, std::regex(regexLine));

            // get the expected values
            std::istringstream valuesStream(valuesLine);
            std::vector<std::string> expectedValues(std::istream_iterator<std::string>{valuesStream}, std::istream_iterator<std::string>());

            ASSERT_EQ(expectedValues.size(), matches.size() - 1) << "the number of expected and found values is different on line (" << lineNumber << ") " << expectedLine << " from " << actualLine;

            // march over each value
            for (int v = 0; v < expectedValues.size(); v++) {
                char compareChar = expectedValues[v][0];
                double expectedValue = expectedValues[v].size() > 1 ? stod(expectedValues[v].substr(1)) : NAN;
                std::string actualValueString = matches[v + 1];

                switch (compareChar) {
                    case '<':
                        ASSERT_LT(std::stod(actualValueString), expectedValue) << " on line " << expectedLine;
                        break;
                    case '>':
                        ASSERT_GT(std::stod(actualValueString), expectedValue) << " on line " << expectedLine;
                        break;
                    case '=':
                        // check some special cases for double values
                        if (std::isnan(expectedValue)) {
                            ASSERT_TRUE(std::isnan(std::stod(actualValueString))) << " on line " << expectedLine;
                        } else {
                            ASSERT_DOUBLE_EQ(std::stod(actualValueString), expectedValue) << " on line " << expectedLine;
                        }
                        break;
                    case '~':
                        // is anything once trimmed
                        ASSERT_TRUE(actualValueString.find_first_not_of(" \t\n\v\f\r") != std::string::npos) << " on line " << expectedLine;
                        break;
                    default:
                        FAIL() << "Unknown compare char " << compareChar << " on line " << expectedLine;
                }
            }

            lineNumber++;
        }
    }

    ASSERT_FALSE(std::getline(actualStream, actualLine)) << "actual results should reach end of file";
}
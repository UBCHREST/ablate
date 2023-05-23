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

void testingResources::MpiTestFixture::CompareOutputFiles() {
    // Compare the output (stdout) file
    if (!mpiTestParameter.expectedOutputFile.empty()) {
        // load in the actual and expected results files
        if (!std::filesystem::exists(mpiTestParameter.expectedOutputFile)) {
            FAIL() << "The expected output file " << mpiTestParameter.expectedOutputFile << " cannot be found";
        }

        CompareOutputFile(mpiTestParameter.expectedOutputFile, OutputFile());
    }

    // compare any other files if provided
    for (const auto& outputFile : mpiTestParameter.expectedFiles) {
        CompareOutputFile(outputFile.first, ResultDirectory() / outputFile.second);
    }
}

static void ExpandExpectedValues(std::vector<std::string>& expectedValues) {
    // Check the expected values to see if they start with a int, if they do expand the results
    for (std::size_t e = 0; e < expectedValues.size(); e++) {
        if (std::isdigit(expectedValues[e][0])) {
            // get the number of times to apply this
            auto split = expectedValues[e].find_first_not_of(" 0123456789");
            int numberOfTimes = std::stoi(expectedValues[e].substr(0, split));
            auto expectedValue = expectedValues[e].substr(split);

            // Delete this component
            expectedValues.erase(expectedValues.begin() + e);

            // replace the values
            for (int n = 0; n < numberOfTimes; n++) {
                expectedValues.insert(expectedValues.begin() + e, expectedValue);
            }
        }
    }
}

void testingResources::MpiTestFixture::CompareOutputFile(const std::string& expectedFileName, const std::string& actualFileName) {
    // make sure we can find both files
    if (!std::filesystem::exists(expectedFileName)) {
        FAIL() << "Cannot locate expectedFile " << expectedFileName;
    }
    if (!std::filesystem::exists(actualFileName)) {
        FAIL() << "Cannot locate actualFile " << actualFileName;
    }

    std::ifstream expectedStream(expectedFileName);
    std::ifstream actualStream(actualFileName);

    // march over each line
    std::string expectedLine;
    std::string actualLine;
    int lineNumber = 1;
    while (std::getline(expectedStream, expectedLine)) {
        if (!std::getline(actualStream, actualLine)) {
            FAIL() << "File " << actualFileName << " is missing lines from " << expectedFileName;
        }

        // check to see if this lines includes any expected values
        auto expectedResultDelimiterPosition = expectedLine.find(expectedResultDelimiter);
        if (expectedResultDelimiterPosition == std::string::npos) {
            // do a direct match
            ASSERT_EQ(expectedLine, actualLine) << " on line(" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
        } else {
            std::string regexLine = expectedLine.substr(0, expectedResultDelimiterPosition);
            std::string valuesLine = expectedLine.substr(expectedResultDelimiterPosition + expectedResultDelimiter.size());

            // get the matches
            std::smatch matches;
            std::vector<std::string> actualValues;

            // Perform multiple searches if needed for each line
            auto searchStart = actualLine.cbegin();
            const auto regex = std::regex(regexLine);

            // while there is more searching
            while (std::regex_search(searchStart, actualLine.cend(), matches, regex)) {
                // If there are any grouped results
                for (size_t i = 1; i < matches.size(); i++) {
                    actualValues.push_back(matches[i]);
                }
                // start the search at the start of the remainder
                searchStart = matches.suffix().first;
            }

            // get the expected values
            std::istringstream valuesStream(valuesLine);
            std::vector<std::string> expectedValues(std::istream_iterator<std::string>{valuesStream}, std::istream_iterator<std::string>());
            ExpandExpectedValues(expectedValues);

            ASSERT_EQ(expectedValues.size(), actualValues.size()) << "the number of expected and found values is different on line (" << lineNumber << ") " << expectedLine << " from " << actualLine
                                                                  << " of file " << expectedFileName;

            // march over each value
            for (std::size_t v = 0; v < expectedValues.size(); v++) {
                char compareChar = expectedValues[v][0];
                double expectedValue = expectedValues[v].size() > 1 ? stod(expectedValues[v].substr(1)) : NAN;
                std::string actualValueString = actualValues[v];

                switch (compareChar) {
                    case '<':
                        ASSERT_LT(std::stod(actualValueString), expectedValue) << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                        break;
                    case '>':
                        ASSERT_GT(std::stod(actualValueString), expectedValue) << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                        break;
                    case '=':
                        // check some special cases for double values
                        if (std::isnan(expectedValue)) {
                            ASSERT_TRUE(std::isnan(std::stod(actualValueString))) << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                        } else {
                            ASSERT_DOUBLE_EQ(std::stod(actualValueString), expectedValue) << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                        }
                        break;
                    case '~':
                        // is any number once trimmed
                        ASSERT_TRUE(actualValueString.find_first_not_of(" \t\n\v\f\r") != std::string::npos) << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                        break;
                    case '*':
                        // is anything of length
                        ASSERT_TRUE(!actualValueString.empty()) << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                        break;
                    case 'z':
                        // should be close to zero
                        ASSERT_LT(std::abs(std::stod(actualValueString)), 1.0E-13) << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                        break;
                    case 'n': {
                        // the value is near percent difference < 1E-3
                        auto percentDifference = PetscAbs((std::stod(actualValueString) - expectedValue) / expectedValue);
                        ASSERT_LT(percentDifference, 1.1E-3) << " the percent difference of (" << expectedValue << ", " << std::stod(actualValueString) << ") should be less than 1E-3 on line "
                                                             << expectedLine << " of file " << expectedFileName;
                    }
                    case 'N': {
                        // the value is near percent difference < 5E-3
                        auto percentDifference = PetscAbs((std::stod(actualValueString) - expectedValue) / expectedValue);
                        ASSERT_LT(percentDifference, 5.1E-3) << " the percent difference of (" << expectedValue << ", " << std::stod(actualValueString) << ") should be less than 1E-3 on line "
                                                             << expectedLine << " of file " << expectedFileName;
                    }break;
                    default:
                        FAIL() << "Unknown compare char " << compareChar << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                }
            }
        }
        lineNumber++;
    }

    ASSERT_FALSE(std::getline(actualStream, actualLine)) << "actual results should reach end of file " << expectedFileName;
}

std::filesystem::path testingResources::MpiTestFixture::BuildResultDirectory() {
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    std::filesystem::path resultDirectory = ResultDirectory();
    if (rank == 0) {
        std::filesystem::remove_all(resultDirectory);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
    return resultDirectory;
}
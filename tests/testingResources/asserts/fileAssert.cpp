#include "fileAssert.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

const std::string expectedResultDelimiter = std::string("<expects>");

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

void testingResources::asserts::FileAssert::CompareFile(const std::filesystem::path& expectedFileName, const std::filesystem::path& actualFileName) {
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
                        auto percentDifference = std::abs((std::stod(actualValueString) - expectedValue) / expectedValue);
                        ASSERT_LT(percentDifference, 1.1E-3) << " the percent difference of (" << expectedValue << ", " << std::stod(actualValueString) << ") should be less than 1E-3 on line "
                                                             << expectedLine << " of file " << expectedFileName;
                    } break;
                    default:
                        FAIL() << "Unknown compare char " << compareChar << " on line (" << lineNumber << ") " << expectedLine << " of file " << expectedFileName;
                }
            }
        }
        lineNumber++;
    }

    ASSERT_FALSE(std::getline(actualStream, actualLine)) << "actual results should reach end of file " << expectedFileName;
}
'
#include "textFileAssert.hpp"
#include "mpiTestFixture.hpp"

#include <utility>
testingResources::asserts::TextFileAssert::TextFileAssert(std::filesystem::path expectedFile, std::string actualFileName)
    : expectedFile(std::move(expectedFile)),
      actualFileName(std::move(actualFileName))

{}

void testingResources::asserts::TextFileAssert::Test(testingResources::MpiTestFixture& mpiTestFixture) {
    // compare any other files if provided
    CompareFile(expectedFile, mpiTestFixture.ResultDirectory() / actualFileName);
}

#include "registrar.hpp"
REGISTER(testingResources::asserts::Assert, testingResources::asserts::TextFileAssert,
         "Compares the expected and actual text files",
         ARG(std::string, "expected", "expectedFile the path to the expected log file (relative to the root of the testing folder)"),
         ARG(std::string, "actual", "the file path inside of the output folder")
         );
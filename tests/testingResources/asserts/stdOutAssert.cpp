#include "stdOutAssert.hpp"
#include "mpiTestFixture.hpp"

#include <utility>
testingResources::asserts::StdOutAssert::StdOutAssert(std::filesystem::path expectedLogFile): expectedLogFile(std::move(expectedLogFile)) {}

void testingResources::asserts::StdOutAssert::Test(testingResources::MpiTestFixture& mpiTestFixture) {
    // Get the log file from the mpiTestFixture
    CompareFile(expectedLogFile, mpiTestFixture.OutputFile());
}


#include "registrar.hpp"
REGISTER_DEFAULT_PASS_THROUGH(testingResources::asserts::Assert, testingResources::asserts::StdOutAssert, "Default assert tests the std out against expected file", std::string);

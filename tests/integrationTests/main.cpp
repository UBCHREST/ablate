#include <gtest/gtest.h>
#include "integrationTest.hpp"
#include "mpiTestEventListener.hpp"
#include "mpiTestFixture.hpp"
#include "yamlParser.hpp"

/**
 * Locate all of the input files that can be used a test.  Any input file without testing parameters or testingIgnore bool will throw an exception.
 * @return
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // register of all tests
    // list all directories input path
    for (const auto& entry : std::filesystem::recursive_directory_iterator("inputs")) {
        // check to see if it is an input file
        if (entry.path().extension() == ".yaml") {
            // Load in the input file
            std::shared_ptr<cppParser::YamlParser> parser = std::make_shared<cppParser::YamlParser>(entry.path());

            // set up the monitor
            if (parser->Contains("test")) {
                auto integrationTest = parser->GetByName<IntegrationTestFixture>("test");

                // Register the test
                integrationTest->RegisterTest(entry);
            } else if (parser->Contains("tests")) {
                auto integrationTests = parser->GetByName<std::vector<IntegrationTestFixture>>("tests");

                // Register the test
                for (auto& integrationTest : integrationTests) {
                    integrationTest->RegisterTest(entry);
                }
            } else if (!parser->GetByName<bool>("testingIgnore", false)) {
                throw std::invalid_argument("An input file " + entry.path().string() + " in integration tests does not contain test parameters.");
            }
        }
    }

    // Store the input parameters
    const bool inMpiTestRun = testingResources::MpiTestFixture::InitializeTestingEnvironment(&argc, &argv);
    if (inMpiTestRun) {
        testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());

        listeners.Append(new testingResources::MpiTestEventListener());
    }

    return RUN_ALL_TESTS();
}
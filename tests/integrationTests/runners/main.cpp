#include <gtest/gtest.h>
#include "mpiTestEventListener.hpp"
#include "mpiTestFixture.hpp"
#include "runners.hpp"

/**
 * This helper lambda gets a list of models from the specified modelPath
 */
std::function<std::vector<IntegrationTestRunnerParameters>()> buildTestParameters = []() {
    std::vector<IntegrationTestRunnerParameters> testParameters;

    // list all directories input path
    for (const auto& entry : std::filesystem::recursive_directory_iterator("inputs")) {
        // check to see if it is an input file
        if (entry.path().extension() == ".yaml") {
            // Load in the input file
            std::shared_ptr<cppParser::YamlParser> parser = std::make_shared<cppParser::YamlParser>(entry.path());

            // set up the monitor
            if(parser->Contains("test")){
                auto mpiParameters = parser->GetByName<testingResources::MpiTestParameter>("test");
                // check to see if it contains a test target
                testParameters.emplace_back(
                    IntegrationTestRunnerParameters{.mpiTestParameter = *mpiParameters,
                                                    .inputFilePath = entry.path()});
            }
            if(parser->Contains("tests")){
                auto mpiParametersList = parser->GetByName<std::vector<testingResources::MpiTestParameter>>("tests");

                for (auto& mpiParameters: mpiParametersList) {
                    // check to see if it contains a test target
                    testParameters.emplace_back(IntegrationTestRunnerParameters{.mpiTestParameter = *mpiParameters, .inputFilePath = entry.path()});
                }
            }
        }
    }

    return testParameters;
};

/**
 * Build the list of test parameters from the provided lambda
 * @return
 */
INSTANTIATE_TEST_SUITE_P(IntergrationTests, IntegrationTestRunnerFixture, testing::ValuesIn(buildTestParameters()), [](const testing::TestParamInfo<IntegrationTestRunnerParameters> &info) {
    return info.param.mpiTestParameter.getTestName();
});

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Store the input parameters
    const bool inMpiTestRun = testingResources::MpiTestFixture::InitializeTestingEnvironment(&argc, &argv);
    if (inMpiTestRun) {
        testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());

        listeners.Append(new testingResources::MpiTestEventListener());
    }

    return RUN_ALL_TESTS();
}
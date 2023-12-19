#ifndef ABLATELIBRARY_INTEGRATIONTEST_HPP
#define ABLATELIBRARY_INTEGRATIONTEST_HPP

#include <cctype>
#include <filesystem>
#include <string>
#include "builder.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "mpiTestFixture.hpp"
#include "mpiTestParamFixture.hpp"
#include "parameters/mapParameters.hpp"
#include "petscsys.h"
#include "yamlParser.hpp"

/**
 * base class for integration type tests.  Actual tests should extend this.  Other types of integration tests should extend this and create their own test class.
 */
class IntegrationTestFixture : public testingResources::MpiTestFixture {
   protected:
    //! baseline mpiTest parameters
    std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter;

    //! store the path to the input file (the file used to generate this.
    std::filesystem::path inputFilePath;

   public:
    explicit IntegrationTestFixture(std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter);

    void SetUp() override { SetMpiParameters(*mpiTestParameter); }

    /**
     * Register this test and test name with the google testing framework
     * @param inputPath
     */
    virtual void RegisterTest(const std::filesystem::path& inputPath) = 0;

   protected:
    //! build and return the test suite name
    [[nodiscard]] std::string GetTestSuiteName(const std::filesystem::path& inputsRoot, const std::string& baseName = "Integration") {
        // Build the test name from the path
        auto relativePath = std::filesystem::relative(inputFilePath.parent_path(), inputsRoot);
        auto relativePathString = relativePath.string();
        if (!relativePathString.empty()) {
            relativePathString[0] = std::toupper(relativePathString[0]);
        }

        auto testSuiteName = baseName + relativePathString;
        auto it = std::remove_if(testSuiteName.begin(), testSuiteName.end(), [](char const& c) { return !std::isalnum(c); });

        testSuiteName.erase(it, testSuiteName.end());
        return testSuiteName;
    }

    //! build and return the test suite name
    [[nodiscard]] std::string GetTestName() {
        auto testName = mpiTestParameter->getTestName();
        if (!testName.empty()) {
            testName[0] = std::toupper(testName[0]);
        }
        return testName;
    }
};

/**
 * The actual class that tests all typical types of integration tests
 */
class IntegrationTest : public IntegrationTestFixture, public std::enable_shared_from_this<IntegrationTest>  {
   public:
    explicit IntegrationTest(std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter);

    /**
     * Code that executes the specific test
     */
    void TestBody() override;

    /**
     * Register this specific test
     * @param inputPath
     */
    void RegisterTest(const std::filesystem::path& inputPath) override;
};

#endif  // ABLATELIBRARY_INTEGRATIONTEST_HPP

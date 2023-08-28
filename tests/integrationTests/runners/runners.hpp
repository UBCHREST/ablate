#ifndef ABLATELIBRARY_RUNNERS_HPP
#define ABLATELIBRARY_RUNNERS_HPP

#include <filesystem>
#include "builder.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "mpiTestFixture.hpp"
#include "mpiTestParamFixture.hpp"
#include "parameters/mapParameters.hpp"
#include "petscsys.h"
#include "yamlParser.hpp"

/**
 * Struct to hold the required information for the integration test runner
 */
struct IntegrationTestRunnerParameters {
    //! baseline mpiTest parameters
    testingResources::MpiTestParameter mpiTestParameter;

    //! store the path to the input file (the file used to generate this.
    std::filesystem::path inputFilePath;

//    //! relative path to the restart input file.  Empty file defaults the original input file
//    std::string restartInputFile;
//
//    //! optional yaml overrides to be passed to the restart input file
//    std::map<std::string, std::string> restartOverrides;
};

class IntegrationTestRunnerFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<IntegrationTestRunnerParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

#endif  // ABLATELIBRARY_RUNNERS_HPP

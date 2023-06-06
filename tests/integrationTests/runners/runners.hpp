#ifndef ABLATELIBRARY_RUNNERS_HPP
#define ABLATELIBRARY_RUNNERS_HPP

#include <filesystem>
#include "MpiTestFixture.hpp"
#include "MpiTestParamFixture.hpp"
#include "builder.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
#include "petscsys.h"
#include "yamlParser.hpp"

/**
 * Note: the test name is assumed to be the relative path to the yaml file
 */
class IntegrationTestsSpecifier : public testingResources::MpiTestParamFixture {};

struct IntegrationRestartTestsParameters {
    //! baseline mpiTest parameters
    testingResources::MpiTestParameter mpiTestParameter;

    //! relative path to the restart input file.  Empty file defaults the original input file
    std::string restartInputFile;

    //! optional yaml overrides to be passed to the restart input file
    std::map<std::string, std::string> restartOverrides;
};

class IntegrationRestartTestsSpecifier : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<IntegrationRestartTestsParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

#endif  // ABLATELIBRARY_RUNNERS_HPP

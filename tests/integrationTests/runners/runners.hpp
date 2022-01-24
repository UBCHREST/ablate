#ifndef ABLATELIBRARY_RUNNERS_HPP
#define ABLATELIBRARY_RUNNERS_HPP

#include <filesystem>
#include <utilities/fileUtility.hpp>
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
    testingResources::MpiTestParameter mpiTestParameter;
    std::map<std::string, std::string> restartOverrides;
};

class IntegrationRestartTestsSpecifier : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<IntegrationRestartTestsParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

#endif  // ABLATELIBRARY_RUNNERS_HPP

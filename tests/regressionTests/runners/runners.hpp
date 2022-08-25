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
class RegressionTestsSpecifier : public testingResources::MpiTestParamFixture {};

#endif  // ABLATELIBRARY_RUNNERS_HPP

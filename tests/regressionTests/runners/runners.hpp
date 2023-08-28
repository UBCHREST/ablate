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
 * Note: the test name is assumed to be the relative path to the yaml file
 */
class RegressionTestsSpecifier : public testingResources::MpiTestParamFixture {};

#endif  // ABLATELIBRARY_RUNNERS_HPP

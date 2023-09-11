#ifndef ABLATELIBRARY_INTEGRATIONRESTARTTEST_HPP
#define ABLATELIBRARY_INTEGRATIONRESTARTTEST_HPP

#include <filesystem>
#include "builder.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "integrationTest.hpp"
#include "mpiTestFixture.hpp"
#include "mpiTestParamFixture.hpp"
#include "parameters/mapParameters.hpp"
#include "petscsys.h"
#include "yamlParser.hpp"

/**
 * Extends the base IntegrationTestFixture and is used to test simulations that need to restart.
 */
class IntegrationRestartTestFixture : public IntegrationTestFixture {
   protected:
    //! relative path to the restart input file.  Empty file defaults the original input file
    std::string restartInputFile;

    //! optional yaml overrides to be passed to the restart input file
    std::shared_ptr<ablate::parameters::Parameters> restartOverrides;

   public:
    explicit IntegrationRestartTestFixture(std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter, std::string restartInputFile,
                                           std::shared_ptr<ablate::parameters::Parameters> restartOverrides
    );
};

/**
 * Integration restart test holds the actual integration restart test code for IntegrationRestartTestFixture
 */
class IntegrationRestartTest : public IntegrationRestartTestFixture {
   public:
    explicit IntegrationRestartTest(std::shared_ptr<testingResources::MpiTestParameter> mpiTestParameter, std::string restartInputFile,
                                    std::shared_ptr<ablate::parameters::Parameters> restartOverrides);

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

#endif  // ABLATELIBRARY_INTEGRATIONRESTARTTEST_HPP

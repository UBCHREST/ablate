#ifndef ABLATELIBRARY_TESTRUNENVIRONMENT_HPP
#define ABLATELIBRARY_TESTRUNENVIRONMENT_HPP

#include <filesystem>
#include "parameters/parameters.hpp"

namespace testingResources {

class TestRunEnvironment {
   public:
    /**
     * Create a environment that resets it after use
     */
    explicit TestRunEnvironment(std::string outputDir = {}, std::string title = "testEnv", bool tagDirectory = false);

    /**
     * Create a environment that resets it after use
     */
    explicit TestRunEnvironment(const ablate::parameters::Parameters& parameters, std::filesystem::path inputPath = {});

    /**
     * clean up the environment
     */
    ~TestRunEnvironment();
};

}  // namespace testingResources

#endif  // ABLATELIBRARY_TEMPORARYPATH_HPP

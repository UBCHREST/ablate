#include "outputDirectory.hpp"
#include <utility>
#include "environment/runEnvironment.hpp"
#include "utilities/temporaryWorkingDirectory.hpp"

ablate::environment::OutputDirectory::OutputDirectory(std::string relativePath) : relativePath(std::move(relativePath)) {}

std::filesystem::path ablate::environment::OutputDirectory::Locate(const std::vector<std::filesystem::path>& searchPaths) {
    return environment::RunEnvironment::Get().GetOutputDirectory() / relativePath;
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(cppParser::PathLocator, ablate::environment::OutputDirectory, "Returns a path relative to the output directory", std::string);

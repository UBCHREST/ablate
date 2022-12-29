#include "download.hpp"
#include <utility>
#include "environment/runEnvironment.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscError.hpp"
#include "utilities/temporaryWorkingDirectory.hpp"

ablate::environment::Download::Download(std::string url) : url(std::move(url)) {}

std::filesystem::path ablate::environment::Download::Locate(const std::vector<std::filesystem::path>& searchPaths) {
    // determine where to download the file
    auto downloadDirectory =
        ablate::environment::RunEnvironment::Get().GetOutputDirectory().empty() ? std::filesystem::current_path() : ablate::environment::RunEnvironment::Get().GetOutputDirectory();

    // Store the current working directory
    ablate::utilities::TemporaryWorkingDirectory temporaryWorkingDirectory(downloadDirectory);

    // check to see if the file specified is really a url
    for (const auto& prefix : urlPrefixes) {
        if (url.rfind(prefix, 0) == 0) {
            char localPath[PETSC_MAX_PATH_LEN];
            PetscBool found;
            PetscFileRetrieve(PETSC_COMM_WORLD, url.c_str(), localPath, PETSC_MAX_PATH_LEN, &found) >> checkError;
            if (!found) {
                throw std::runtime_error("unable to locate file at" + url);
            }

            // If we should relocate the file
            return downloadDirectory / localPath;
        }
    }
    throw std::invalid_argument("Unknown url scheme " + url);
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(cppParser::PathLocator, ablate::environment::Download, "Downloads and relocates file at given url", std::string);

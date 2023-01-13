#include "gitHub.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include "environment/runEnvironment.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/temporaryWorkingDirectory.hpp"

ablate::environment::GitHub::GitHub(std::string repository, std::string path, std::string gitHubToken) : repository(repository), path(path), gitHubToken(gitHubToken) {}

std::filesystem::path ablate::environment::GitHub::Download(std::string urlPath, std::filesystem::path localPath, bool root) {
    // check to see if the file exists
    // this is a directory
    std::filesystem::path newPath = localPath / std::filesystem::path(urlPath).filename();
    if (exists(newPath)) {
        return newPath;
    }

    // Store the current working directory
    ablate::utilities::TemporaryWorkingDirectory temporaryWorkingDirectory(localPath);

    // download the metadata for this file
    auto metaDataUrl = urlBase + repository;
    metaDataUrl += contentSeperator;
    metaDataUrl += urlPath;
    if (!gitHubToken.empty()) {
        metaDataUrl += " -H 'Authorization: token ";
        metaDataUrl += gitHubToken;
        metaDataUrl += "'";
    }
    metaDataUrl += " -H ignore:/";
    metaDataUrl += std::filesystem::path(urlPath).filename();
    metaDataUrl += ".json";

    // download the file
    char meatDataFilePath[PETSC_MAX_PATH_LEN];
    PetscBool found;
    PetscFileRetrieve(PETSC_COMM_WORLD, metaDataUrl.c_str(), meatDataFilePath, PETSC_MAX_PATH_LEN, &found) >> utilities::PetscUtilities::checkError;
    if (!found) {
        throw std::runtime_error("unable to locate metadata for " + urlPath + " in repo " + repository);
    }

    // check the file size
    if (std::filesystem::is_empty(meatDataFilePath)) {
        throw std::runtime_error("unable to locate metadata for " + urlPath + " in repo " + repository);
    }

    // parse the metadata
    std::ifstream metaDataFile;
    metaDataFile.open(meatDataFilePath);
    nlohmann::json metaData;
    metaDataFile >> metaData;
    metaDataFile.close();
    // remote the metadata file
    MPI_Barrier(PETSC_COMM_WORLD);
    if (root) {
        std::filesystem::remove(meatDataFilePath);
    }

    // check to see if the response is for a directory
    if (metaData.is_array()) {
        if (root) {
            // store the new created path
            create_directories(newPath);
        }
        MPI_Barrier(PETSC_COMM_WORLD);

        // Now march over each possible directory/file
        for (const auto& object : metaData) {
            Download(object["path"], newPath, root);
        }
    } else {
        // download the file
        std::string downloadUrl = metaData["download_url"];

        // append the name
        downloadUrl += " -H ignore:/";
        downloadUrl += metaData["name"];

        PetscFileRetrieve(PETSC_COMM_SELF, downloadUrl.c_str(), meatDataFilePath, PETSC_MAX_PATH_LEN, &found) >> utilities::PetscUtilities::checkError;
        if (!found) {
            throw std::runtime_error("unable to locate file at " + downloadUrl);
        }
    }

    // set path the currentWorkingDirectory
    return newPath;
}

std::filesystem::path ablate::environment::GitHub::Locate(const std::vector<std::filesystem::path>&) {
    // Get the current rank
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> utilities::MpiUtilities::checkError;

    std::filesystem::path localPath =
        ablate::environment::RunEnvironment::Get().GetOutputDirectory().empty() ? std::filesystem::temp_directory_path() / path : ablate::environment::RunEnvironment::Get().GetOutputDirectory();
    if (ablate::environment::RunEnvironment::Get().GetOutputDirectory().empty() && std::filesystem::exists(localPath)) {
        std::filesystem::remove_all(localPath);
    }
    if (rank == 0) {
        std::filesystem::create_directories(localPath);
    }
    MPI_Barrier(PETSC_COMM_WORLD);

    // call recursive download function
    return Download(path, localPath, rank == 0);
}

#include "registrar.hpp"
REGISTER(cppParser::PathLocator, ablate::environment::GitHub, "Downloads file/directory from github repo", ARG(std::string, "repository", "repository to download file from (must include owner/name)"),
         ARG(std::string, "path", "the relative path in the repo"),
         OPT(std::string, "token",
             "optional github [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) to access repo or bypass rate "
             "limits."));

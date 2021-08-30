#include "fileUtility.hpp"
#include <petscsys.h>
#include "mpiError.hpp"
#include "petscError.hpp"

std::filesystem::path ablate::utilities::FileUtility::LocateFile(std::string file, MPI_Comm com, std::vector<std::filesystem::path> searchPaths, std::filesystem::path remoteRelocatePath) {
    // check to see if the path exists
    if (std::filesystem::exists(file)) {
        return std::filesystem::path(file);
    }

    // check to see if the file specified is really a url
    for (const auto& prefix : urlPrefixes) {
        if (file.rfind(prefix, 0) == 0) {
            char localPath[PETSC_MAX_PATH_LEN];
            PetscBool found;
            PetscFileRetrieve(PETSC_COMM_WORLD, file.c_str(), localPath, PETSC_MAX_PATH_LEN, &found) >> checkError;
            if (!found) {
                throw std::runtime_error("unable to locate file at" + file);
            }

            // If we should relocate the file
            if (!remoteRelocatePath.empty()) {
                // Get the current rank
                PetscMPIInt rank;
                MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> checkMpiError;

                // create a new path (this assumes same file system on all machines)
                auto newPath = remoteRelocatePath / std::filesystem::path(localPath).filename();
                if (rank == 0) {
                    std::filesystem::copy(localPath, newPath, std::filesystem::copy_options::overwrite_existing);
                }
                MPI_Barrier(PETSC_COMM_WORLD);

                return newPath;
            } else {
                return localPath;
            }
        }
    }

    // check for the file in local search directories
    for (const auto& directory : searchPaths) {
        // build a test path
        auto testPath = directory / file;
        if (std::filesystem::exists(testPath)) {
            return testPath;
        }
    }

    throw std::runtime_error("unable to locate file " + file);
}

#ifndef ABLATELIBRARY_FILEUTILITY_HPP
#define ABLATELIBRARY_FILEUTILITY_HPP

#include <petsc.h>
#include <filesystem>
#include <string>
#include <vector>

namespace ablate::utilities {
class FileUtility {
   private:
    const MPI_Comm comm;
    const std::vector<std::filesystem::path> searchPaths;
    const std::filesystem::path remoteRelocatePath;

    // include list of prefixes for urls
    inline static std::vector<std::string> urlPrefixes = {"https://", "http://"};

   public:
    explicit FileUtility(MPI_Comm comm = MPI_COMM_SELF, std::vector<std::filesystem::path> searchPaths = {}, std::filesystem::path remoteRelocatePath = {});

    std::filesystem::path Locate(std::string name);

    std::function<std::filesystem::path(std::string)> GetLocateFileFunction();

    /**
     * public static call to be used without a class instance
     * @param name
     * @param com
     * @param searchPaths
     * @param remoteRelocatePath
     * @return
     */
    static std::filesystem::path LocateFile(std::string name, MPI_Comm comm = MPI_COMM_SELF, std::vector<std::filesystem::path> searchPaths = {}, std::filesystem::path remoteRelocatePath = {});
};
};      // namespace ablate::utilities
#endif  // ABLATELIBRARY_FILEUTILITY_HPP

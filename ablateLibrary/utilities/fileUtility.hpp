#ifndef ABLATELIBRARY_FILEUTILITY_HPP
#define ABLATELIBRARY_FILEUTILITY_HPP

#include <petsc.h>
#include <filesystem>
#include <string>
#include <vector>

namespace ablate::utilities {
class FileUtility {
   private:
    FileUtility() = delete;

    // include list of prefixes for urls
    inline static std::vector<std::string> urlPrefixes = {"https://", "http://"};

   public:
    static std::filesystem::path LocateFile(std::string name, MPI_Comm com = MPI_COMM_SELF, std::vector<std::filesystem::path> searchPaths = {}, std::filesystem::path remoteRelocatePath = {});
};
};      // namespace ablate::utilities
#endif  // ABLATELIBRARY_FILEUTILITY_HPP

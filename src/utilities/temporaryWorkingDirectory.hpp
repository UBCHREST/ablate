#ifndef ABLATELIBRARY_TEMPORARYWORKINGDIRECTORY_HPP
#define ABLATELIBRARY_TEMPORARYWORKINGDIRECTORY_HPP

#include <filesystem>

namespace ablate::utilities {

/**
 * this class changes the working directory as specified until it is destructed.  At this point it changes it back.
 */
class TemporaryWorkingDirectory {
   private:
    const std::filesystem::path currentWorkingDirectory;

   public:
    TemporaryWorkingDirectory(const std::filesystem::path& tempWorkingDirectory) : currentWorkingDirectory(std::filesystem::current_path()) { std::filesystem::current_path(tempWorkingDirectory); }
    ~TemporaryWorkingDirectory() { std::filesystem::current_path(currentWorkingDirectory); }
};

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_TEMPORARYWORKINGDIRECTORY_HPP

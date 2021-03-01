#ifndef ABLATELIBRARY_PETSCOPTIONS_HPP
#define ABLATELIBRARY_PETSCOPTIONS_HPP
#include <map>
#include <string>

namespace ablate::utilities {
class PetscOptions {
   public:
    static void Set(const std::string& prefix, const std::map<std::string, std::string>& options);
    static void Set(const std::map<std::string, std::string>& options);
};
}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_PETSCOPTIONS_HPP

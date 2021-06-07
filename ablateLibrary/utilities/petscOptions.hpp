#ifndef ABLATELIBRARY_PETSCOPTIONS_HPP
#define ABLATELIBRARY_PETSCOPTIONS_HPP
#include <petsc.h>
#include <map>
#include <string>

namespace ablate::utilities {
class PetscOptionsUtils {
   public:
    static void Set(const std::string& prefix, const std::map<std::string, std::string>& options);
    static void Set(const std::map<std::string, std::string>& options);
    static void Set(PetscOptions petscOptions, const std::map<std::string, std::string>& options);
};

void PetscOptionsDestroyAndCheck(std::string name, PetscOptions *options);

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_PETSCOPTIONS_HPP

//
// Created by Matt McGurn on 2/15/21.
//

#ifndef ABLATELIBRARY_PETSCOPTIONS_HPP
#define ABLATELIBRARY_PETSCOPTIONS_HPP
#include <map>
#include <string>

namespace ablate {
namespace utilities {
class PetscOptions {
   public:
    static void Set(const std::string& prefix, const std::map<std::string, std::string>& options);
    static void Set(const std::map<std::string, std::string>& options);
};
}
}

#endif  // ABLATELIBRARY_PETSCOPTIONS_HPP

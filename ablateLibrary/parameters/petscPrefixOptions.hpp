#ifndef ABLATELIBRARY_PETSCPREFIXOPTIONS_HPP
#define ABLATELIBRARY_PETSCPREFIXOPTIONS_HPP

#include "mapParameters.hpp"
namespace ablate::parameters {

class PetscPrefixOptions : public MapParameters {
   private:
    std::string prefix;

   public:
    explicit PetscPrefixOptions(std::string prefix);
    ~PetscPrefixOptions() = default;
};
}

#endif  // ABLATELIBRARY_PETSCPREFIXOPTIONS_HPP
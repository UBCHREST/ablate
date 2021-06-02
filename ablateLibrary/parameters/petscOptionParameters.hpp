#ifndef ABLATELIBRARY_PETSCOPTIONPARAMETERS_HPP
#define ABLATELIBRARY_PETSCOPTIONPARAMETERS_HPP
#include "parameters.hpp"
#include <petsc.h>

namespace ablate::parameters {

class PetscOptionParameters :public Parameters {
   private:
    const PetscOptions petscOptions;

   public:
    PetscOptionParameters(PetscOptions petscOptions = nullptr);
    ~PetscOptionParameters() = default;

    std::optional<std::string> GetString(std::string paramName) const override;
    std::unordered_set<std::string> GetKeys() const override;
};

}

#endif  // ABLATELIBRARY_PETSCOPTIONPARAMETERS_HPP

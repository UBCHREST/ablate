#ifndef ABLATELIBRARY_PETSCOPTIONPARAMETERS_HPP
#define ABLATELIBRARY_PETSCOPTIONPARAMETERS_HPP
#include <petsc.h>
#include "parameters.hpp"

namespace ablate::parameters {

class PetscOptionParameters : public Parameters {
   protected:
    PetscOptions petscOptions;

   public:
    PetscOptionParameters(PetscOptions petscOptions = nullptr);
    virtual ~PetscOptionParameters() = default;

    std::optional<std::string> GetString(std::string paramName) const override;
    std::unordered_set<std::string> GetKeys() const override;
};

}  // namespace ablate::parameters

#endif  // ABLATELIBRARY_PETSCOPTIONPARAMETERS_HPP

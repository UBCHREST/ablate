#ifndef ABLATELIBRARY_RBF_PHS_HPP
#define ABLATELIBRARY_RBF_PHS_HPP

#include "rbf.hpp"

#define __RBF_PHS_DEFAULT_PARAM 5

namespace ablate::domain::rbf {

class PHS : public RBF {
   private:
    const PetscInt phsOrder = -1;  // The PHS order

   public:
    std::string_view type() const override { return "PHS"; }

    PHS(PetscInt p = 4, PetscInt phsOrder = 4, bool doesNotHaveDerivatives = false, bool doesNotHaveInterpolation = false);

    PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;
};

}  // namespace ablate::domain::rbf

#endif  // ABLATELIBRARY_RBF_PHS_HPP

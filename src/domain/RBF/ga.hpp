#ifndef ABLATELIBRARY_RBF_GA_HPP
#define ABLATELIBRARY_RBF_GA_HPP

#include "rbf.hpp"

#define __RBF_GA_DEFAULT_PARAM 0.1

namespace ablate::domain::rbf {

class GA : public RBF {
   private:
    const PetscReal scale = 0.1;

   public:
    std::string_view type() const override { return "GA"; }

    GA(PetscInt p = 4, PetscReal scale = 0.1, bool doesNotHaveDerivatives = false, bool doesNotHaveInterpolation = false);

    PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;
};

}  // namespace ablate::domain::rbf

#endif  // ABLATELIBRARY_RBF_GA_HPP

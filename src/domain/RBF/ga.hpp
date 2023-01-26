#ifndef ABLATELIBRARY_RBF_GA_HPP
#define ABLATELIBRARY_RBF_GA_HPP

#include "rbf.hpp"

#define __RBF_GA_DEFAULT_PARAM 0.1

namespace ablate::domain::rbf {

class GA: public RBF {
  private:
    const PetscReal scale = 0.1;

  public:

    std::string_view type() const override { return "GA"; }

    GA(PetscInt p = 4, PetscReal scale = 0.1, bool hasDerivatives = false, bool hasInterpolation = false);

    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;

};

}  // namespace ablate::domain::RBF

#endif  // ABLATELIBRARY_RBF_GA_HPP

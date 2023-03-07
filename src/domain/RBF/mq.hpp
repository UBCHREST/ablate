#ifndef ABLATELIBRARY_RBF_MQ_HPP
#define ABLATELIBRARY_RBF_MQ_HPP

#include "rbf.hpp"

#define __RBF_MQ_DEFAULT_PARAM 0.1

namespace ablate::domain::rbf {

class MQ: virtual public RBF {
  private:
    const PetscReal scale = -1;

  public:

    std::string_view type() const override { return "MQ"; }

    MQ(PetscInt p = 4, PetscReal scale = 0.1, bool doesNotHaveDerivatives = false, bool doesNotHaveInterpolation = false);

    PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;

};


}  // namespace ablate::domain::RBF

#endif  // ABLATELIBRARY_RBF_MQ_HPP

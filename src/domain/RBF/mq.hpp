#ifndef ABLATELIBRARY_RBF_MQ_HPP
#define ABLATELIBRARY_RBF_MQ_HPP

#include "rbf.hpp"

#define __RBF_MQ_DEFAULT_PARAM 0.1

namespace ablate::domain::rbf {

class MQ: virtual public RBF {
  private:
    const PetscReal scale = -1;

  public:
    MQ(PetscInt p = 4, PetscReal scale = 0.1, bool hasDerivatives = false, bool hasInterpolation = false);

    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;

};


}  // namespace ablate::domain::RBF

#endif  // ABLATELIBRARY_RBF_MQ_HPP

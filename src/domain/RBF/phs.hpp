#ifndef ABLATELIBRARY_RBF_PHS_HPP
#define ABLATELIBRARY_RBF_PHS_HPP

#include "rbf.hpp"

#define __RBF_PHS_DEFAULT_PARAM 9

namespace ablate::domain::rbf {

class PHS : public RBF {
  private:
    const PetscInt  phsOrder = -1;    // The PHS order

  public:
    PHS(PetscInt p = 4, PetscInt phsOrder = 4, bool hasDerivatives = false, bool hasInterpolation = false);

    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;


};

}  // namespace ablate::domain::RBF

#endif  // ABLATELIBRARY_RBF_PHS_HPP

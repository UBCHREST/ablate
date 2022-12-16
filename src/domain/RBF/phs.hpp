#ifndef ABLATELIBRARY_RBF_PHS_HPP
#define ABLATELIBRARY_RBF_PHS_HPP

#include "rbf.hpp"

namespace ablate::domain::rbf {

class PHS : public RBF {
  private:
    const PetscInt  phsOrder = -1;    // The PHS order

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    PHS(PetscInt p = 4, PetscInt phsOrder = 4, bool hasDerivatives = false, bool hasInterpolation = false);

    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }

};

}  // namespace ablate::domain::RBF

#endif  // ABLATELIBRARY_RBF_PHS_HPP

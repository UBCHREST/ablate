#ifndef ABLATELIBRARY_RBF_IMQ_HPP
#define ABLATELIBRARY_RBF_IMQ_HPP

#include "rbf.hpp"

namespace ablate::domain::rbf {

class IMQ: public RBF {
  private:
    const PetscReal scale = -1;

    PetscReal InternalVal(PetscReal x[], PetscReal y[]);
    PetscReal InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz);
  public:
    IMQ(PetscInt p = 4, PetscReal scale = 0.1, bool hasDerivatives = false, bool hasInterpolation = false);

    PetscReal RBFVal(PetscReal x[], PetscReal y[]) override {return InternalVal(std::move(x), std::move(y)); }
    PetscReal RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override {return InternalDer(std::move(x), std::move(dx), std::move(dy), std::move(dz)); }
};

}  // namespace ablate::domain::RBF

#endif  // ABLATELIBRARY_RBF_IMQ_HPP

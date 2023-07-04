#ifndef ABLATELIBRARY_RBF_PHS_HPP
#define ABLATELIBRARY_RBF_PHS_HPP

#include "rbf.hpp"

#define __RBF_PHS_DEFAULT_PARAM 5

namespace ablate::domain::rbf {

class PHS : public RBF {
   private:
    const int phsOrder = -1;  // The PHS order

   public:
    std::string_view type() const override { return "PHS"; }

    PHS(int p = 4, int phsOrder = 4, bool doesNotHaveDerivatives = false, bool doesNotHaveInterpolation = false, bool returnNeighborVertices = false);

    PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;
};

}  // namespace ablate::domain::rbf

#endif  // ABLATELIBRARY_RBF_PHS_HPP

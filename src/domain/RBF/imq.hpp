#ifndef ABLATELIBRARY_RBF_IMQ_HPP
#define ABLATELIBRARY_RBF_IMQ_HPP

#include "rbf.hpp"

#define __RBF_IMQ_DEFAULT_PARAM 0.1

namespace ablate::domain::rbf {

class IMQ : public RBF {
   private:
    const double scale = -1;

   public:
    std::string_view type() const override { return "IMQ"; }

    IMQ(int p = 4, double scale = 0.1, bool doesNotHaveDerivatives = false, bool doesNotHaveInterpolation = false, bool useNeighborVertices = false);

    PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;
};

}  // namespace ablate::domain::rbf

#endif  // ABLATELIBRARY_RBF_IMQ_HPP

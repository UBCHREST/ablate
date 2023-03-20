#ifndef ABLATELIBRARY_RBF_HYBRID_HPP
#define ABLATELIBRARY_RBF_HYBRID_HPP

// A hybrid RBF method based on weighted sums of individual kernels. See "A stabalized radial basis-finite difference (RBF-FD) mthod with hybrid kernels"
//  by Mishra, Fasshauer, Sen, and Ling, Computer and Mathematics with Applications 77 (2019) 2354-2369.
//
// Notes: The sub-kernel parameters polyOrder, hasDerivative, and hasInterpolation are ignored/not used.

#include "rbf.hpp"

namespace ablate::domain::rbf {

class HYBRID : public RBF {
   private:
    const std::vector<double> weights;
    std::vector<std::shared_ptr<RBF>> rbfList;

   public:
    std::string_view type() const override { return "HYBRID"; }

    HYBRID(int p = 4, std::vector<double> weights = {}, std::vector<std::shared_ptr<RBF>> rbfList = {}, bool doesNotHaveDerivatives = false, bool doesNotHaveInterpolation = false);

    PetscReal RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) override;
    PetscReal RBFDer(PetscInt dim, PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) override;
};

}  // namespace ablate::domain::rbf

#endif  // ABLATELIBRARY_RBF_HYBRID_HPP

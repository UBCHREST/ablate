#include "hybrid.hpp"

#include "imq.hpp"

using namespace ablate::domain::rbf;

/************ Begin Polyharmonic Spline Derived Class **********************/
HYBRID::HYBRID(PetscInt p, std::vector<double> weights, std::vector<std::shared_ptr<RBF>> rbfList, bool doesNotHaveDerivatives, bool doesNotHaveInterpolation)
    : RBF(p, !doesNotHaveDerivatives, !doesNotHaveInterpolation), weights(weights), rbfList(rbfList){};

// Polyharmonic spline: r^(2*m+1)
PetscReal HYBRID::RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) {
    PetscReal val = 0.0;
    for (long unsigned int i = 0; i < HYBRID::rbfList.size(); ++i) {
        val += (HYBRID::weights[i]) * (HYBRID::rbfList[i]->RBFVal(dim, x, y));
    }
    return val;
}

// Derivatives of Polyharmonic spline at a location.
PetscReal HYBRID::RBFDer(PetscInt dim, PetscReal xIn[], PetscInt dx, PetscInt dy, PetscInt dz) {
    PetscReal val = 0.0;
    for (long unsigned int i = 0; i < HYBRID::rbfList.size(); ++i) {
        val += (HYBRID::weights[i]) * (HYBRID::rbfList[i]->RBFDer(dim, xIn, dx, dy, dz));
    }
    return val;
}

/************ End Polyharmonic Spline Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::HYBRID, "Radial Basis Function",
         OPT(PetscInt, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= dim. Any value <dim will result in a polyOrder of 4."),
         ARG(std::vector<double>, "weights", "The scale to apply to each sub-RBF. This must match the order in which they are defined."),
         ARG(std::vector<ablate::domain::rbf::RBF>, "rbfList", "List of RBF kernels to use."), OPT(bool, "doesNotHaveDerivatives", "Compute derivative information. Default is false."),
         OPT(bool, "doesNotHaveInterpolation", "Compute interpolation information. Default is false."));

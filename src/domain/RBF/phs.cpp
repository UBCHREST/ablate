#include "phs.hpp"

using namespace ablate::domain::rbf;

/************ Begin Polyharmonic Spline Derived Class **********************/
PHS::PHS(PetscInt p, PetscInt phsOrder, bool doesNotHaveDerivatives, bool doesNotHaveInterpolation)
    : RBF(p, !doesNotHaveDerivatives, !doesNotHaveInterpolation), phsOrder(phsOrder < 1 ? __RBF_PHS_DEFAULT_PARAM : 2 * phsOrder + 1){};

// Polyharmonic spline: r^(2*m+1)
PetscReal PHS::RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) {
    PetscReal r = PetscSqrtReal(PHS::DistanceSquared(dim, x, y));
    PetscReal phs = PetscPowRealInt(r, PHS::phsOrder);

    return phs;
}

// Derivatives of Polyharmonic spline at a location.
PetscReal PHS::RBFDer(PetscInt dim, PetscReal xIn[], PetscInt dx, PetscInt dy, PetscInt dz) {
    const PetscInt m = PHS::phsOrder;
    const PetscReal r2 = PHS::DistanceSquared(dim, xIn);
    const PetscReal r = PetscSqrtReal(r2);
    PetscReal phs;
    PetscReal x[3];

    PHS::Loc3D(dim, xIn, x);

    switch (dx + 10 * dy + 100 * dz) {
        case 0:
            phs = PetscPowRealInt(r, m);
            break;
        case 1:  // x
            phs = -m * x[0] * PetscPowRealInt(r, m - 2);
            break;
        case 2:  // xx
            phs = m * ((m - 1) * x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) * PetscPowRealInt(r, m - 4);
            break;
        case 10:  // y
            phs = -m * x[1] * PetscPowRealInt(r, m - 2);
            break;
        case 20:  // yy
            phs = m * (x[0] * x[0] + (m - 1) * x[1] * x[1] + x[2] * x[2]) * PetscPowRealInt(r, m - 4);
            break;
        case 100:  // z
            phs = -m * x[2] * PetscPowRealInt(r, m - 2);
            break;
        case 200:  // zz
            phs = m * (x[0] * x[0] + x[1] * x[1] + (m - 1) * x[2] * x[2]) * PetscPowRealInt(r, m - 4);
            break;
        case 11:  // xy
            phs = m * (m - 2) * x[0] * x[1] * PetscPowRealInt(r, m - 4);
            break;
        case 101:  // xz
            phs = m * (m - 2) * x[0] * x[2] * PetscPowRealInt(r, m - 4);
            break;
        case 110:  // yz
            phs = m * (m - 2) * x[1] * x[2] * PetscPowRealInt(r, m - 4);
            break;
        case 111:  // xyz
            phs = -m * (m - 2) * (m - 4) * x[0] * x[1] * x[2] * PetscPowRealInt(r, m - 6);
            break;
        default:
            throw std::invalid_argument("PHS: Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
    }

    return phs;
}

/************ End Polyharmonic Spline Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::PHS, "Radial Basis Function",
         OPT(PetscInt, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= dim. Any value <dim will result in a polyOrder of 4."),
         OPT(PetscInt, "phsOrder", "Order of the polyharmonic spline. Must be >=1. The actual order will be 2*phsOrder+1. Any value <1 will result in a default order of 9."),
         OPT(bool, "doesNotHaveDerivatives", "Compute derivative information. Default is false."), OPT(bool, "doesNotHaveInterpolation", "Compute interpolation information. Default is false."));

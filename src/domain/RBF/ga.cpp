#include "ga.hpp"

using namespace ablate::domain::rbf;

/************ Begin Gaussian Derived Class **********************/
GA::GA(int p, double scale, bool doesNotHaveDerivatives, bool doesNotHaveInterpolation, bool returnNeighborVertices)
    : RBF(p, !doesNotHaveDerivatives, !doesNotHaveInterpolation, returnNeighborVertices), scale(scale < PETSC_SMALL ? __RBF_GA_DEFAULT_PARAM : scale){};

// Gaussian: r^m
PetscReal GA::RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) {
    PetscReal h = GA::scale;
    PetscReal e2 = 1.0 / (h * h);
    PetscReal r2 = GA::DistanceSquared(dim, x, y);

    return PetscExpReal(-r2 * e2);
}

// Derivatives of Gaussian spline at a location.
PetscReal GA::RBFDer(PetscInt dim, PetscReal xIn[], PetscInt dx, PetscInt dy, PetscInt dz) {
    PetscReal h = GA::scale;
    PetscReal e2 = 1.0 / (h * h);
    PetscReal r2 = GA::DistanceSquared(dim, xIn);
    PetscReal ga = PetscExpReal(-r2 * e2);
    PetscReal x[3];

    GA::Loc3D(dim, xIn, x);

    switch (dx + 10 * dy + 100 * dz) {
        case 0:
            // Do nothing
            break;
        case 1:  // x
            ga *= 2.0 * e2 * x[0];
            break;
        case 2:  // xx
            ga *= 2.0 * e2 * (2.0 * e2 * x[0] * x[0] - 1.0);
            break;
        case 10:  // y
            ga *= 2.0 * e2 * x[1];
            break;
        case 20:  // yy
            ga *= 2.0 * e2 * (2.0 * e2 * x[1] * x[1] - 1.0);
            break;
        case 100:  // z
            ga *= 2.0 * e2 * x[2];
            break;
        case 200:  // zz
            ga *= 2.0 * e2 * (2.0 * e2 * x[2] * x[2] - 1.0);
            break;
        case 11:  // xy
            ga *= 4.0 * e2 * e2 * x[0] * x[1];
            break;
        case 101:  // xz
            ga *= 4.0 * e2 * e2 * x[0] * x[2];
            break;
        case 110:  // yz
            ga *= 4.0 * e2 * e2 * x[1] * x[2];
            break;
        case 111:  // xyz
            ga *= 8.0 * e2 * e2 * e2 * x[0] * x[1] * x[2];
            break;
        default:
            throw std::invalid_argument("GA: Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
    }

    return ga;
}

/************ End Gaussian Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::GA, "Radial Basis Function",
         OPT(int, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= 1. Any value <1 will result in a polyOrder of 4."),
         OPT(double, "scale", "Scaling parameter. Must be >0. Any value <PETSC_SMALL will result in a default scale of 0.1."),
         OPT(bool, "doesNotHaveDerivatives", "Compute derivative information. Default is false."),
         OPT(bool, "doesNotHaveInterpolation", "Compute interpolation information. Default is false."),
         OPT(bool, "returnNeighborVertices", "Perform RBF based on neighboring vertices (TRUE) or cells (FALSE). Default is false."));

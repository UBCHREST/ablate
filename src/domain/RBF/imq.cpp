#include "imq.hpp"

using namespace ablate::domain::rbf;

/************ Begin Inverse Multiquadric Derived Class **********************/
IMQ::IMQ(int p, double scale, bool doesNotHaveDerivatives, bool doesNotHaveInterpolation, bool useNeighborVertices)
    : RBF(p, !doesNotHaveDerivatives, !doesNotHaveInterpolation, useNeighborVertices), scale(scale < PETSC_SMALL ? __RBF_IMQ_DEFAULT_PARAM : scale){};

// Inverse Multiquadric: 1/sqrt(1+(er)^2)
PetscReal IMQ::RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) {
    PetscReal h = IMQ::scale;
    PetscReal e2 = 1.0 / (h * h);
    PetscReal r2 = IMQ::DistanceSquared(dim, x, y);

    return 1.0 / PetscSqrtReal(1.0 + e2 * r2);
}

// Derivatives of Inverse Multiquadric spline at a location.
PetscReal IMQ::RBFDer(PetscInt dim, PetscReal xIn[], PetscInt dx, PetscInt dy, PetscInt dz) {
    PetscReal h = IMQ::scale;
    PetscReal e2 = 1.0 / (h * h);
    PetscReal r2 = IMQ::DistanceSquared(dim, xIn);
    PetscReal imq = PetscSqrtReal(1.0 + e2 * r2);
    PetscReal x[3];

    IMQ::Loc3D(dim, xIn, x);

    switch (dx + 10 * dy + 100 * dz) {
        case 0:
            imq = 1.0 / imq;
            break;
        case 1:  // x
            imq = e2 * x[0] / (imq * imq * imq);
            break;
        case 2:  // xx
            imq = -e2 * (1.0 + e2 * (-2.0 * x[0] * x[0] + x[1] * x[1] + x[2] * x[2])) / (imq * imq * imq * imq * imq);
            break;
        case 10:  // y
            imq = e2 * x[1] / (imq * imq * imq);
            break;
        case 20:  // yy
            imq = -e2 * (1.0 + e2 * (x[0] * x[0] - 2.0 * x[1] * x[1] + x[2] * x[2])) / (imq * imq * imq * imq * imq);
            break;
        case 100:  // z
            imq = e2 * x[2] / (imq * imq * imq);
            break;
        case 200:  // zz
            imq = -e2 * (1.0 + e2 * (x[0] * x[0] + x[1] * x[1] - 2.0 * x[2] * x[2])) / (imq * imq * imq * imq * imq);
            break;
        case 11:  // xy
            imq = 3.0 * PetscSqr(e2) * x[0] * x[1] / (imq * imq * imq * imq * imq);
            break;
        case 101:  // xz
            imq = 3.0 * PetscSqr(e2) * x[0] * x[2] / (imq * imq * imq * imq * imq);
            break;
        case 110:  // yz
            imq = 3.0 * PetscSqr(e2) * x[1] * x[2] / (imq * imq * imq * imq * imq);
            break;
        case 111:  // xyz
            imq = 15.0 * e2 * PetscSqr(e2) * x[0] * x[1] * x[2] / (imq * imq * imq * imq * imq * imq * imq);
            break;
        default:
            throw std::invalid_argument("IMQ: Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
    }

    return imq;
}

/************ End Inverse Multiquadric Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::IMQ, "Radial Basis Function",
         OPT(int, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= 1. Any value <1 will result in a polyOrder of 4."),
         OPT(double, "scale", "Scaling parameter. Must be >0. Any value <PETSC_SMALL will result in a default scale of 0.1."),
         OPT(bool, "doesNotHaveDerivatives", "Compute derivative information. Default is false."), OPT(bool, "doesNotHaveInterpolation", "Compute interpolation information. Default is false."),
         OPT(bool, "useNeighborVertices", "Perform RBF based on neighboring vertices (TRUE) or cells (FALSE). Default is false."));

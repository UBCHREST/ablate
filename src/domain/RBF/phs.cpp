#include "phs.hpp"

using namespace ablate::domain::rbf;

/************ Begin Polyharmonic Spline Derived Class **********************/
PHS::PHS(PetscInt p, PetscInt phsOrder, bool hasDerivatives, bool hasInterpolation) :
  RBF(p, hasDerivatives, hasInterpolation),
  phsOrder(phsOrder < 1 ? __RBF_PHS_DEFAULT_PARAM : phsOrder) {};


 //Polyharmonic spline: r^(2*m+1)
PetscReal PHS::RBFVal(PetscReal x[], PetscReal y[]) {
  PetscReal r = PetscSqrtReal(PHS::DistanceSquared(x, y));
  PetscReal phs = PetscPowRealInt(r, 2*(PHS::phsOrder) + 1);

  return phs;
}

// Derivatives of Polyharmonic spline at a location.
PetscReal PHS::RBFDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {
  const PetscInt m = PHS::phsOrder;
  const PetscReal r2 = PHS::DistanceSquared(x);
  const PetscReal r = PetscSqrtReal(r2);
  PetscReal phs;

  switch (dx + 10*dy + 100*dz) {
    case 0:
      phs = PetscPowRealInt(r, 2*(PHS::phsOrder) + 1);
      break;
    case 1: // x
      phs = r*(1+2*m)*x[0]*PetscPowRealInt(r2, m-1);
      break;
    case 2: // xx
      phs = r*(1+2*m)*(2*m*x[0]*x[0]+x[1]*x[1]+x[2]*x[2])*PetscPowRealInt(r2, m-2);
      break;
    case 10: // y
      phs = r*(1+2*m)*x[1]*PetscPowRealInt(r2, m-1);
      break;
    case 20: // yy
      phs = r*(1+2*m)*(x[0]*x[0]+2*m*x[1]*x[1]+x[2]*x[2])*PetscPowRealInt(r2, m-2);
      break;
    case 100: // z
      phs = r*(1+2*m)*x[2]*PetscPowRealInt(r2, m-1);
      break;
    case 200: // zz
      phs = r*(1+2*m)*(x[0]*x[0]+x[1]*x[1]+2*m*x[2]*x[2])*PetscPowRealInt(r2, m-2);
      break;
    case 11: // xy
      phs = r*(4*m*m-1)*x[0]*x[1]*PetscPowRealInt(r2, m-2);
      break;
    case 101: // xz
      phs = r*(4*m*m-1)*x[0]*x[2]*PetscPowRealInt(r2, m-2);
      break;
    case 110: // yz
      phs = r*(4*m*m-1)*x[1]*x[2]*PetscPowRealInt(r2, m-2);
      break;
    case 111: // xyz
      phs = r*(2*m-3)*(2*m-1)*(2*m+1)*x[0]*x[1]*x[2]*PetscPowRealInt(r2, m-3);
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
         OPT(PetscInt, "phsOrder", "Order of the polyharmonic spline. Must be >=1. Any value <1 will result in a default order of 4."),
         OPT(bool, "hasDerivatives", "Compute derivative information. Default is false."),
         OPT(bool, "hasInterpolation", "Compute interpolation information. Default is false.")
         );

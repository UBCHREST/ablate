#include "phs.hpp"

using namespace ablate::domain::rbf;

/************ Begin Polyharmonic Spline Derived Class **********************/
PHS::PHS(PetscInt p, PetscInt phsOrder, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), phsOrder(phsOrder) {};


 //Polyharmonic spline: r^m
PetscReal PHS::InternalVal(PetscReal x[], PetscReal y[]) {
  PetscInt  m = PHS::phsOrder;   // The PHS order
  PetscReal r = PHS::DistanceSquared(x, y);

  return PetscPowReal(r, 0.5*((PetscReal)m));
}

// Derivatives of Polyharmonic spline at a location.
PetscReal PHS::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {
  PetscInt  m = PHS::phsOrder;   // The PHS order
  PetscReal r = PHS::DistanceSquared(x);

  r = PetscSqrtReal(r);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      r = PetscPowReal(r, (PetscReal)m);
      break;
    case 1: // x
      r = -m*x[0]*PetscPowReal(r, (PetscReal)(m-2));
      break;
    case 2: // xx
      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[0]*x[0]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 10: // y
      r = -m*x[1]*PetscPowReal(r, (PetscReal)(m-2));
      break;
    case 20: // yy
      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[1]*x[1]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 100: // z
      r = -m*x[2]*PetscPowReal(r, (PetscReal)(m-2));
      break;
    case 200: // zz
      r = m*PetscPowReal(r, (PetscReal)(m-2)) + m*(m-2)*x[2]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 11: // xy
      r = m*(m-2)*x[0]*x[1]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 101: // xz
      r = m*(m-2)*x[0]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    case 110: // yz
      r = m*(m-2)*x[1]*x[2]*PetscPowReal(r, (PetscReal)(m-4));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
  }

  return r;
}
/************ End Polyharmonic Spline Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::PHS, "Radial Basis Function",
         OPT(PetscInt, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= 1. Default is 4."),
         OPT(PetscInt, "phsOrder", "Order of the polyharmonic spline. Must be >=1. Default is 4."),
         OPT(bool, "hasDerivatives", "Compute derivative information. Default is false."),
         OPT(bool, "hasInterpolation", "Compute interpolation information. Default is false.")
         );

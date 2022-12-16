#include "imq.hpp"


using namespace ablate::domain::rbf;

/************ Begin Inverse Multiquadric Derived Class **********************/
IMQ::IMQ(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};


// Multiquadric: sqrt(1+(er)^2)
PetscReal IMQ::InternalVal(PetscReal x[], PetscReal y[]) {

  PetscReal h = IMQ::scale;
  PetscReal e = 1.0/h;
  PetscReal r = IMQ::DistanceSquared(x, y);

  return 1.0/PetscSqrtReal(1.0 + e*e*r);
}

// Derivatives of Multiquadric spline at a location.
PetscReal IMQ::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

  PetscReal h = IMQ::scale;
  PetscReal e = 1.0/h;
  PetscReal r = IMQ::DistanceSquared(x);

  r = PetscSqrtReal(1.0 + e*e*r);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      r = 1.0/r;
      break;
    case 1: // x
      r = -e*e*x[0]/PetscPowReal(r, 3.0);
      break;
    case 2: // xx
      r = -e*e*(1.0 + e*e*(-2.0*x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 5.0);
      break;
    case 10: // y
      r = -e*e*x[1]/PetscPowReal(r, 3.0);
      break;
    case 20: // yy
      r = -e*e*(1.0 + e*e*(x[0]*x[0] - 2.0*x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 5.0);
      break;
    case 100: // z
      r = -e*e*x[2]/PetscPowReal(r, 3.0);
      break;
    case 200: // zz
      r = -e*e*(1.0 + e*e*(x[0]*x[0] + x[1]*x[1] - 2.0*x[2]*x[2]))/PetscPowReal(r, 5.0);
      break;
    case 11: // xy
      r = 3.0*PetscSqr(e*e)*x[0]*x[1]/PetscPowReal(r, 5.0);
      break;
    case 101: // xz
      r = 3.0*PetscSqr(e*e)*x[0]*x[2]/PetscPowReal(r, 5.0);
      break;
    case 110: // yz
      r = 3.0*PetscSqr(e*e)*x[1]*x[2]/PetscPowReal(r, 5.0);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
  }

  return r;
}
/************ End Inverse Multiquadric Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::IMQ, "Radial Basis Function",
         OPT(PetscInt, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= 1. Default is 4."),
         OPT(PetscReal, "scale", "Scaling parameter. Default is 0.1."),
         OPT(bool, "hasDerivatives", "Compute derivative information. Default is false."),
         OPT(bool, "hasInterpolation", "Compute interpolation information. Default is false.")
         );

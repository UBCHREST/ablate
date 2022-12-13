#include "ga.hpp"


using namespace ablate::domain::rbf;

/************ Begin Gaussian Derived Class **********************/
GA::GA(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt p, PetscReal scale) : RBF(subDomain, p), scale(scale) {};


// Gaussian: r^m
PetscReal GA::InternalVal(PetscReal x[], PetscReal y[]) {

  PetscReal h = GA::scale;
  PetscReal e2 = 1.0/(h*h);
  PetscReal r = GA::DistanceSquared(x, y);

  return PetscExpReal(-r*e2);
}

// Derivatives of Gaussian spline at a location.
PetscReal GA::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

  PetscReal h = GA::scale;
  PetscReal e2 = 1.0/(h*h);
  PetscReal r = GA::DistanceSquared(x);

  r = PetscExpReal(-r*e2);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      // Do nothing
      break;
    case 1: // x
      r *= -2.0*e2*x[0];
      break;
    case 2: // xx
      r *= 2.0*e2*(2.0*e2*x[0]*x[0]-1.0);
      break;
    case 10: // x[1]
      r *= -2.0*e2*x[1];
      break;
    case 20: // yy
      r *= 2.0*e2*(2.0*e2*x[1]*x[1]-1.0);
      break;
    case 100: // x[2]
      r *= -2.0*e2*x[2];
      break;
    case 200: // zz
      r *= 2.0*e2*(2.0*e2*x[2]*x[2]-1.0);
      break;
    case 11: // xy
      r *= 4.0*e2*e2*x[0]*x[1];
      break;
    case 101: // xz
      r *= 4.0*e2*e2*x[0]*x[2];
      break;
    case 110: // yz
      r *= 4.0*e2*e2*x[1]*x[2];
      break;
    case 111:
      r *= 8.0*e2*e2*e2*x[1]*x[2];
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown derivative!\n");
  }

  return r;
}
/************ End Gaussian Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::GA, "Radial Basis Function",
         ARG(ablate::domain::SubDomain , "subDomain", "The sub-domain to use."),
         OPT(PetscInt, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= 1. Default is 4."),
         OPT(PetscReal, "scale", "Scaling parameter. Default is 0.1.")
         );



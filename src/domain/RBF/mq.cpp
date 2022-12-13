#include "mq.hpp"

using namespace ablate::domain::rbf;

/************ Begin Multiquadric Derived Class **********************/
MQ::MQ(std::shared_ptr<ablate::domain::SubDomain> subDomain, PetscInt p, PetscReal scale) : RBF(subDomain, p), scale(scale) {

  printf("IN MQ!\n");
  PetscFinalize();
  };


// Multiquadric: sqrt(1+(er)^2)
PetscReal MQ::InternalVal(PetscReal x[], PetscReal y[]) {

  PetscReal h = MQ::scale;
  PetscReal e = 1.0/h;
  PetscReal r = MQ::DistanceSquared(x, y);

  return PetscSqrtReal(1.0 + e*e*r);
}

// Derivatives of Multiquadric spline at a location.
PetscReal MQ::InternalDer(PetscReal x[], PetscInt dx, PetscInt dy, PetscInt dz) {

  PetscReal h = MQ::scale;
  PetscReal e = 1.0/h;
  PetscReal r = MQ::DistanceSquared(x);

  r = PetscSqrtReal(1.0 + e*e*r);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      // Do nothing
      break;
    case 1: // x
      r = -e*e*x[0]/r;
      break;
    case 2: // xx
      r = e*e*(1.0 + e*e*(x[1]*x[1] + x[2]*x[2]))/PetscPowReal(r, 3.0);
      break;
    case 10: // y
      r = -e*e*x[1]/r;
      break;
    case 20: // yy
      r = e*e*(1.0 + e*e*(x[0]*x[0] + x[2]*x[2]))/PetscPowReal(r, 3.0);
      break;
    case 100: // z
      r = -e*e*x[2]/r;
      break;
    case 200: // zz
      r = e*e*(1.0 + e*e*(x[0]*x[0] + x[1]*x[1]))/PetscPowReal(r, 3.0);
      break;
    case 11: // xy
      r = -PetscSqr(e*e)*x[0]*x[1]/PetscPowReal(r, 3.0);
      break;
    case 101: // xz
      r = -PetscSqr(e*e)*x[0]*x[2]/PetscPowReal(r, 3.0);
      break;
    case 110: // yz
      r = -PetscSqr(e*e)*x[1]*x[2]/PetscPowReal(r, 3.0);
      break;
    default:
      throw std::invalid_argument("Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
  }

  return r;
}

/************ End Multiquadric Derived Class **********************/

//#include "registrar.hpp"
//REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::MQ, "Radial Basis Function",
//         ARG(ablate::domain::SubDomain , "subDomain", "The sub-domain to use."),
//         OPT(PetscInt, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= 1. Default is 4."),
//         OPT(PetscReal, "scale", "Scaling parameter. Default is 0.1.")
//         );

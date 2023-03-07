#include "mq.hpp"

using namespace ablate::domain::rbf;

/************ Begin Multiquadric Derived Class **********************/

MQ::MQ(PetscInt p, PetscReal scale, bool doesNotHaveDerivatives, bool doesNotHaveInterpolation) :
  RBF(p, !doesNotHaveDerivatives, !doesNotHaveInterpolation),
  scale(scale < PETSC_SMALL ? __RBF_MQ_DEFAULT_PARAM : scale) {};


// Multiquadric: sqrt(1+(er)^2)
PetscReal MQ::RBFVal(PetscInt dim, PetscReal x[], PetscReal y[]) {

  PetscReal h = MQ::scale;
  PetscReal e2 = 1.0/(h*h);
  PetscReal r2 = MQ::DistanceSquared(dim, x, y);

  return PetscSqrtReal(1.0 + e2*r2);
}

// Derivatives of Multiquadric spline at a location.
PetscReal MQ::RBFDer(PetscInt dim, PetscReal xIn[], PetscInt dx, PetscInt dy, PetscInt dz) {

  PetscReal h = MQ::scale;
  PetscReal e2 = 1.0/(h*h);
  PetscReal r2 = MQ::DistanceSquared(dim, xIn);
  PetscReal mq = PetscSqrtReal(1.0 + e2*r2);
  PetscReal x[3];

  MQ::Loc3D(dim, xIn, x);

  switch (dx + 10*dy + 100*dz) {
    case 0:
      // Do nothing
      break;
    case 1: // x
      mq = -e2*x[0]/mq;
      break;
    case 2: // xx
      mq = e2*(1.0 + e2*(x[1]*x[1] + x[2]*x[2]))/(mq*mq*mq);
      break;
    case 10: // y
      mq = -e2*x[1]/mq;
      break;
    case 20: // yy
      mq = e2*(1.0 + e2*(x[0]*x[0] + x[2]*x[2]))/(mq*mq*mq);
      break;
    case 100: // z
      mq = -e2*x[2]/mq;
      break;
    case 200: // zz
      mq = e2*(1.0 + e2*(x[0]*x[0] + x[1]*x[1]))/(mq*mq*mq);
      break;
    case 11: // xy
      mq = -PetscSqr(e2)*x[0]*x[1]/(mq*mq*mq);
      break;
    case 101: // xz
      mq = -PetscSqr(e2)*x[0]*x[2]/(mq*mq*mq);
      break;
    case 110: // yz
      mq = -PetscSqr(e2)*x[1]*x[2]/(mq*mq*mq);
      break;
    case 111: // xyz
      mq = -3.0*e2*PetscSqr(e2)*x[0]*x[1]*x[2]/(mq*mq*mq*mq*mq);
      break;
    default:
      throw std::invalid_argument("MQ: Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
  }

  return mq;
}

/************ End Multiquadric Derived Class **********************/

#include "registrar.hpp"
REGISTER(ablate::domain::rbf::RBF, ablate::domain::rbf::MQ, "Radial Basis Function",
         OPT(PetscInt, "polyOrder", "Order of the augmenting RBF polynomial. Must be >= 1. Any value <1 will result in a polyOrder of 4."),
         OPT(PetscReal, "scale", "Scaling parameter. Must be >0. Any value <PETSC_SMALL will result in a default scale of 0.1."),
         OPT(bool, "doesNotHaveDerivatives", "Compute derivative information. Default is false."),
         OPT(bool, "doesNotHaveInterpolation", "Compute interpolation information. Default is false.")
         );

#include "rbf.hpp"


using namespace ablate::domain::rbf;

void RBF::Normal1D(const ablate::domain::Field *field, const PetscInt c, PetscScalar *n) {

  PetscReal cx = 0.0, g = 0.0;

  cx = RBF::EvalDer(field, c, 1, 0, 0);
  g = PetscSqrtReal(cx*cx);

  n[0] = cx/g;
}

void RBF::Normal2D(const ablate::domain::Field *field, const PetscInt c, PetscScalar *n) {

  PetscReal   cx = 0.0, cy = 0.0, g = 0.0;

  cx = RBF::EvalDer(field, c, 1, 0, 0);
  cy = RBF::EvalDer(field, c, 0, 1, 0);
  g = PetscSqrtReal(cx*cx + cy*cy);

  n[0] = cx/g;
  n[1] = cy/g;


}

void RBF::Normal3D(const ablate::domain::Field *field, const PetscInt c, PetscReal *n) {

  PetscReal   cx = 0.0, cy = 0.0, cz = 0.0, g = 0.0;

  cx = RBF::EvalDer(field, c, 1, 0, 0);
  cy = RBF::EvalDer(field, c, 0, 1, 0);
  cz = RBF::EvalDer(field, c, 0, 0, 1);
  g = sqrt(cx*cx + cy*cy + cz*cz);

  n[0] = cx/g;
  n[1] = cy/g;
  n[2] = cz/g;
}

void RBF::Normal(const ablate::domain::Field *field, const PetscInt c, PetscReal *n) {
  switch (RBF::subDomain->GetDimensions()) {
    case 1:
      RBF::Normal1D(field, c, n);
      break;
    case 2:
      RBF::Normal2D(field, c, n);
      break;
    case 3:
      RBF::Normal3D(field, c, n);
      break;
    default:
      throw std::runtime_error("ablate::levelSet::geometry::Normal encountered an unknown dimension.");
  }
}

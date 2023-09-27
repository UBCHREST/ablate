
#include "geometry.hpp"
#include <string>
#include <vector>
#include "domain/RBF/rbf.hpp"


// Might want to have this based on cell values rather than RBF
static void Normal1D(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c, PetscScalar *n) {

  PetscReal cx = 0.0, g = 0.0;

  cx = rbf->EvalDer(field, c, 1, 0, 0);
  g = PetscSqrtReal(cx*cx);

  n[0] = cx/g;
}

static void Normal2D(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c, PetscScalar *n) {

  PetscReal   cx = 0.0, cy = 0.0, g = 0.0;

  cx = rbf->EvalDer(field, c, 1, 0, 0);
  cy = rbf->EvalDer(field, c, 0, 1, 0);
  g = PetscSqrtReal(cx*cx + cy*cy);

  n[0] = cx/g;
  n[1] = cy/g;


}

static void Normal3D(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c, PetscReal *n) {

  PetscReal   cx = 0.0, cy = 0.0, cz = 0.0, g = 0.0;

  cx = rbf->EvalDer(field, c, 1, 0, 0);
  cy = rbf->EvalDer(field, c, 0, 1, 0);
  cz = rbf->EvalDer(field, c, 0, 0, 1);
  g = sqrt(cx*cx + cy*cy + cz*cz);

  n[0] = cx/g;
  n[1] = cy/g;
  n[2] = cz/g;
}

static PetscReal Curvature2D(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c) {

  PetscReal k = 0.0;
  PetscReal cx, cy, cxx, cyy, cxy;

  cx = rbf->EvalDer(field, c, 1, 0, 0);
  cy = rbf->EvalDer(field, c, 0, 1, 0);
  cxx = rbf->EvalDer(field, c, 2, 0, 0);
  cyy = rbf->EvalDer(field, c, 0, 2, 0);
  cxy = rbf->EvalDer(field, c, 1, 1, 0);

  k = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/pow(cx*cx+cy*cy,1.5);

  return k;
}

static PetscReal Curvature3D(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c) {

  PetscReal k = 0.0;
  PetscReal cx, cy, cz;
  PetscReal cxx, cyy, czz;
  PetscReal cxy, cxz, cyz;

  cx = rbf->EvalDer(field, c, 1, 0, 0);
  cy = rbf->EvalDer(field, c, 0, 1, 0);
  cz = rbf->EvalDer(field, c, 0, 0, 1);
  cxx = rbf->EvalDer(field, c, 2, 0, 0);
  cyy = rbf->EvalDer(field, c, 0, 2, 0);
  czz = rbf->EvalDer(field, c, 0, 0, 2);
  cxy = rbf->EvalDer(field, c, 1, 1, 0);
  cxz = rbf->EvalDer(field, c, 1, 0, 1);
  cyz = rbf->EvalDer(field, c, 0, 1, 1);

  k = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2.0*(cxy*cx*cy + cxz*cx*cz + cyz*cy*cz))/pow(cx*cx+cy*cy+cz*cz,1.5);

  return k;
}


PetscReal ablate::levelSet::geometry::Curvature(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c) {
  switch (rbf->GetDimensions()) {
    case 1:
      return 0.0;
    case 2:
      return Curvature2D(rbf, field, c);
    case 3:
      return Curvature3D(rbf, field, c);
    default:
      throw std::runtime_error("ablate::levelSet::geometry::Curvature encountered an unknown dimension.");
  }
}

void ablate::levelSet::geometry::Normal(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c, PetscReal *n) {
  switch (rbf->GetDimensions()) {
    case 1:
      Normal1D(rbf, field, c, n);
      break;
    case 2:
      Normal2D(rbf, field, c, n);
      break;
    case 3:
      Normal3D(rbf, field, c, n);
      break;
    default:
      throw std::runtime_error("ablate::levelSet::geometry::Normal encountered an unknown dimension.");
  }
}

#include "rbf.hpp"

using namespace ablate::domain::rbf;

PetscReal RBF::Curvature2D(const ablate::domain::Field *field, const PetscInt c) {

  PetscReal k = 0.0;
  PetscReal cx, cy, cxx, cyy, cxy;

  cx = RBF::EvalDer(field, c, 1, 0, 0);
  cy = RBF::EvalDer(field, c, 0, 1, 0);
  cxx = RBF::EvalDer(field, c, 2, 0, 0);
  cyy = RBF::EvalDer(field, c, 0, 2, 0);
  cxy = RBF::EvalDer(field, c, 1, 1, 0);

  k = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/pow(cx*cx+cy*cy,1.5);

  return k;
}

PetscReal RBF::Curvature3D(const ablate::domain::Field *field, const PetscInt c) {

  PetscReal k = 0.0;
  PetscReal cx, cy, cz;
  PetscReal cxx, cyy, czz;
  PetscReal cxy, cxz, cyz;

  cx = RBF::EvalDer(field, c, 1, 0, 0);
  cy = RBF::EvalDer(field, c, 0, 1, 0);
  cz = RBF::EvalDer(field, c, 0, 0, 1);
  cxx = RBF::EvalDer(field, c, 2, 0, 0);
  cyy = RBF::EvalDer(field, c, 0, 2, 0);
  czz = RBF::EvalDer(field, c, 0, 0, 2);
  cxy = RBF::EvalDer(field, c, 1, 1, 0);
  cxz = RBF::EvalDer(field, c, 1, 0, 1);
  cyz = RBF::EvalDer(field, c, 0, 1, 1);

  k = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2.0*(cxy*cx*cy + cxz*cx*cz + cyz*cy*cz))/pow(cx*cx+cy*cy+cz*cz,1.5);

  return k;
}


PetscReal RBF::Curvature(const ablate::domain::Field *field, const PetscInt c) {
  switch (RBF::subDomain->GetDimensions()) {
    case 1:
      return 0.0;
    case 2:
      return RBF::Curvature2D(field, c);
    case 3:
      return RBF::Curvature3D(field, c);
    default:
      throw std::runtime_error("ablate::levelSet::geometry::Curvature encountered an unknown dimension.");
  }
}

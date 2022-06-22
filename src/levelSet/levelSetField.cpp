#include "levelSetField.hpp"


using namespace ablate::levelSet;


PetscReal LevelSetField::Curvature2D(PetscInt c){

  PetscReal                       k = 0.0;
  PetscReal                       cx, cy, cxx, cyy, cxy;
  Vec                             phi = LevelSetField::phi;
  std::shared_ptr<DerCalculator>  der = LevelSetField::der;

  cx = der->EvalDer(phi, c, 1, 0, 0);
  cy = der->EvalDer(phi, c, 0, 1, 0);
  cxx = der->EvalDer(phi, c, 2, 0, 0);
  cyy = der->EvalDer(phi, c, 0, 2, 0);
  cxy = der->EvalDer(phi, c, 1, 1, 0);

  k = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/pow(cx*cx+cy*cy,1.5);

  return k;
}

PetscReal LevelSetField::Curvature3D(PetscInt c){

  PetscReal                       k = 0.0;
  PetscReal                       cx, cy, cz;
  PetscReal                       cxx, cyy, czz;
  PetscReal                       cxy, cxz, cyz;
  Vec                             phi = LevelSetField::phi;
  std::shared_ptr<DerCalculator>  der = LevelSetField::der;

  cx = der->EvalDer(phi, c, 1, 0, 0);
  cy = der->EvalDer(phi, c, 0, 1, 0);
  cz = der->EvalDer(phi, c, 0, 0, 1);
  cxx = der->EvalDer(phi, c, 2, 0, 0);
  cyy = der->EvalDer(phi, c, 0, 2, 0);
  czz = der->EvalDer(phi, c, 0, 0, 2);
  cxy = der->EvalDer(phi, c, 1, 1, 0);
  cxz = der->EvalDer(phi, c, 1, 0, 1);
  cyz = der->EvalDer(phi, c, 0, 1, 1);

  k = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2.0*(cxy*cx*cy + cxz*cx*cz + cyz*cy*cz))/pow(cx*cx+cy*cy+cz*cz,1.5);

  return k;
}

LevelSetField::LevelSetField(std::shared_ptr<domain::Region> region) : region(region) {}

LevelSetField::LevelSetField(std::shared_ptr<RBF> rbf, LevelSetField::levelSetShape shape) {

  LevelSetField::rbf = rbf;
  LevelSetField::dm = rbf->GetDM();

  DMCreateGlobalVector(dm, &(LevelSetField::phi)) >> ablate::checkError;

  PetscInt            d, dim;
  PetscInt            cStart, cEnd, c;
  PetscInt            cnt;
  PetscInt            nSet = 3;
  PetscScalar         *val;
  PetscReal           lo[] = {0.0, 0.0, 0.0}, hi[] = {0.0, 0.0, 0.0}, centroid[] = {0.0, 0.0, 0.0}, pos[] = {0.0, 0.0, 0.0};
  PetscReal           radius = 1.0;
  Vec                 phi = LevelSetField::phi;


  DMGetDimension(dm, &dim) >> ablate::checkError;
  DMGetBoundingBox(dm, lo, hi) >> ablate::checkError;
  for (d = 0; d < dim; ++d) {
    centroid[d] = 0.5*(lo[d] + hi[d]);
  }

  PetscOptionsGetReal(NULL, NULL, "-radius", &(radius), NULL) >> ablate::checkError;
  PetscOptionsGetRealArray(NULL, NULL, "-centroid", centroid, &nSet, NULL) >> ablate::checkError;


  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::checkError;       // Range of cells
  VecGetArray(phi, &val) >> ablate::checkError;


  // Set the initial shape.
// I don't know if the global section is the best way to do it. It works, but need to ask Matt K. about it.
  PetscSection  section;
  PetscInt      gdof;
  DMGetGlobalSection(dm, &section) >> ablate::checkError;
  cnt = 0;

  switch (shape) {
    case LevelSetField::levelSetShape::SPHERE:
     for (c = cStart; c < cEnd; ++c) {
        PetscSectionGetDof(section, c, &gdof) >> ablate::checkError;
        if (gdof > 0) {
          DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
          val[cnt++] = LevelSetField::Sphere(pos, centroid, radius);
        }
      }

      break;
    case LevelSetField::levelSetShape::ELLIPSE:
      for (c = cStart; c < cEnd; ++c) {
        PetscSectionGetDof(section, c, &gdof) >> ablate::checkError;
        if (gdof > 0) {
          DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
          val[cnt++] = LevelSetField::Ellipse(pos, centroid, radius);
        }
      }

      break;
    case LevelSetField::levelSetShape::STAR:
      for (c = cStart; c < cEnd; ++c) {
        PetscSectionGetDof(section, c, &gdof) >> ablate::checkError;
        if (gdof > 0) {
          DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
          val[cnt++] = LevelSetField::Star(pos, centroid);
        }
      }

      break;
    default:
      throw std::invalid_argument("Unknown level set shape shape");
  }

  VecRestoreArray(phi, &val) >> ablate::checkError;

  VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD);

  // Now setup the derivatives and set the curvature/normal calculations
  PetscInt nDer = 0;
  PetscInt *dx, *dy, *dz;

  int i = 0;
  if (dim == 2) {
    nDer = 5;
    PetscMalloc3(nDer, &dx, nDer, &dy, nDer, &dz);

    dx[i] = 1; dy[i] = 0; dz[i++] = 0;
    dx[i] = 0; dy[i] = 1; dz[i++] = 0;
    dx[i] = 2; dy[i] = 0; dz[i++] = 0;
    dx[i] = 0; dy[i] = 2; dz[i++] = 0;
    dx[i] = 1; dy[i] = 1; dz[i++] = 0;

//    LevelSetField::Curvature = LevelSetField::Curvature2D;

  }
  else {
    nDer = 10;
    PetscMalloc3(nDer, &dx, nDer, &dy, nDer, &dz);

    dx[i] = 1; dy[i] = 0; dz[i++] = 0;
    dx[i] = 0; dy[i] = 1; dz[i++] = 0;
    dx[i] = 0; dy[i] = 0; dz[i++] = 1;

    dx[i] = 2; dy[i] = 0; dz[i++] = 0;
    dx[i] = 0; dy[i] = 2; dz[i++] = 0;
    dx[i] = 0; dy[i] = 0; dz[i++] = 2;

    dx[i] = 1; dy[i] = 1; dz[i++] = 0;
    dx[i] = 1; dy[i] = 0; dz[i++] = 1;
    dx[i] = 0; dy[i] = 1; dz[i++] = 1;

    dx[i] = 1; dy[i] = 1; dz[i++] = 1;
  }

  auto der = std::make_shared<ablate::levelSet::DerCalculator>(rbf, nDer, dx, dy, dz);



  PetscFree(dx);
  PetscFree(dy);
  PetscFree(dz);


}

LevelSetField::~LevelSetField() {
  if (LevelSetField::phi) {
    VecDestroy(&(LevelSetField::phi));
  }
  if (der) {
    der->~DerCalculator();
  }
}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> LevelSetField::GetFields() {
    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> levelSetField{
        std::make_shared<domain::FieldDescription>("level set field", "phi", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM, region)};

    return levelSetField;
}



/* Sphere */
PetscReal LevelSetField::Sphere(PetscReal pos[], PetscReal center[], PetscReal radius) {
  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
  PetscReal phi = PetscSqrtReal(PetscSqr(shiftedPos[0]) + PetscSqr(shiftedPos[1]) + PetscSqr(shiftedPos[2])) - radius;
  return phi;
}

/* Ellipse */
PetscReal LevelSetField::Ellipse(PetscReal pos[], PetscReal center[], PetscReal radius) {
  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
  PetscReal phi = PetscSqr(shiftedPos[0]/0.5) + PetscSqr(shiftedPos[1]/1.25) + PetscSqr(shiftedPos[2]) - radius;
  return phi;
}


/* Star */
PetscReal LevelSetField::Star(PetscReal pos[], PetscReal center[]) {
  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
  PetscReal phi = 400.0*shiftedPos[0]*shiftedPos[0]*shiftedPos[1]*shiftedPos[1]-(1.0-0.5*shiftedPos[0]*shiftedPos[0]-0.5*shiftedPos[1]*shiftedPos[1]);
  return phi;
}




#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, LevelSetField, "Level Set fields need for interface tracking",
         OPT(ablate::domain::Region, "region", "the region for the level set (defaults to entire domain)"));

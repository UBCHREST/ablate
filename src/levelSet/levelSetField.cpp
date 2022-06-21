#include "levelSetField.hpp"


using namespace ablate::levelSet;


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

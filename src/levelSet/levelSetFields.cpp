#include "levelSetFields.hpp"
#include "domain/fieldDescription.hpp"

ablate::levelSet::LevelSetFields::LevelSetFields(std::shared_ptr<domain::Region> region) : region(region) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::levelSet::LevelSetFields::GetFields() {
    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> lsFields{
        std::make_shared<domain::FieldDescription>(LEVELSET_FIELD, LEVELSET_FIELD, domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM, region),
        std::make_shared<domain::FieldDescription>(NORMAL_FIELD, NORMAL_FIELD, std::vector<std::string>{NORMAL_FIELD + domain::FieldDescription::DIMENSION}, domain::FieldLocation::AUX, domain::FieldType::FVM, region),
        std::make_shared<domain::FieldDescription>(CURVATURE_FIELD, CURVATURE_FIELD, domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::AUX, domain::FieldType::FVM, region)};

    return lsFields;
}


#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::levelSet::LevelSetFields, "Level set fields need for interface tracking",
         OPT(ablate::domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"));



































//#include "levelSetField.hpp"


//using namespace ablate::levelSet;

//LevelSetField::LevelSetField(std::shared_ptr<domain::Region> region) : region(region) {}




////LevelSetField::LevelSetField(std::shared_ptr<ablate::radialBasis::RBF> rbf, LevelSetField::levelSetShape shape) {

////  LevelSetField::rbf = rbf;
////  LevelSetField::dm = rbf->GetDM();

////  PetscInt            d, dim;
////  PetscInt            cStart, cEnd, c;
////  PetscInt            nSet = 3;
////  PetscScalar         *val;
////  PetscReal           lo[] = {0.0, 0.0, 0.0}, hi[] = {0.0, 0.0, 0.0}, centroid[] = {0.0, 0.0, 0.0}, pos[] = {0.0, 0.0, 0.0};
////  PetscReal           radius = 1.0;



////  DMGetDimension(dm, &dim) >> ablate::checkError;
////  LevelSetField::dim = dim;




////  // Create the vectors
////  DMCreateGlobalVector(dm, &(LevelSetField::phi)) >> ablate::checkError;
////  DMCreateGlobalVector(dm, &(LevelSetField::curv)) >> ablate::checkError;

////  PetscInt lsz, gsz;
////  VecGetLocalSize(phi, &lsz) >> ablate::checkError;
////  VecGetSize(phi, &gsz) >> ablate::checkError;
////  VecCreateMPI(PETSC_COMM_WORLD, dim*lsz, dim*gsz, &(LevelSetField::normal)) >> ablate::checkError;
////  VecSetBlockSize(LevelSetField::normal, dim) >> ablate::checkError;

////  DMGetBoundingBox(dm, lo, hi) >> ablate::checkError;
////  for (d = 0; d < dim; ++d) {
////    centroid[d] = 0.5*(lo[d] + hi[d]);
////  }

////  PetscOptionsGetReal(NULL, NULL, "-radius", &(radius), NULL) >> ablate::checkError;
////  PetscOptionsGetRealArray(NULL, NULL, "-centroid", centroid, &nSet, NULL) >> ablate::checkError;


////  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::checkError;       // Range of cells
////  VecGetArray(LevelSetField::phi, &val) >> ablate::checkError;
////  // Set the initial shape.
////  switch (shape) {
////    case LevelSetField::levelSetShape::SPHERE:
////     for (c = cStart; c < cEnd; ++c) {
////       DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
////       val[c - cStart] = LevelSetField::Sphere(pos, centroid, radius);
////      }

////      break;
////    case LevelSetField::levelSetShape::ELLIPSE:
////      for (c = cStart; c < cEnd; ++c) {
////        DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
////        val[c - cStart] = LevelSetField::Ellipse(pos, centroid, radius);
////      }

////      break;
////    case LevelSetField::levelSetShape::STAR:
////      for (c = cStart; c < cEnd; ++c) {
////        DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
////        val[c - cStart] = LevelSetField::Star(pos, centroid);
////      }

////      break;
////    default:
////      throw std::invalid_argument("Unknown level set shape shape");
////  }

////  VecRestoreArray(LevelSetField::phi, &val) >> ablate::checkError;

////  VecGhostUpdateBegin(LevelSetField::phi, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;
////  VecGhostUpdateEnd(LevelSetField::phi, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;

////  // Now setup the derivatives and set the curvature/normal calculations
////  PetscInt nDer = 0;
////  PetscInt dx[10], dy[10], dz[10];

////  nDer = ( dim == 2 ) ? 5 : 10;
////  PetscInt i = 0;
////  dx[i] = 1; dy[i] = 0; dz[i++] = 0;
////  dx[i] = 0; dy[i] = 1; dz[i++] = 0;
////  dx[i] = 2; dy[i] = 0; dz[i++] = 0;
////  dx[i] = 0; dy[i] = 2; dz[i++] = 0;
////  dx[i] = 1; dy[i] = 1; dz[i++] = 0;
////  if( dim == 3) {
////    dx[i] = 0; dy[i] = 0; dz[i++] = 1;
////    dx[i] = 0; dy[i] = 0; dz[i++] = 2;
////    dx[i] = 1; dy[i] = 0; dz[i++] = 1;
////    dx[i] = 0; dy[i] = 1; dz[i++] = 1;
////    dx[i] = 1; dy[i] = 1; dz[i++] = 1;
////  }
////  rbf->SetDerivatives(nDer, dx, dy, dz);

////}

//LevelSetField::~LevelSetField() {
//  VecDestroy(&(LevelSetField::phi));
//  VecDestroy(&(LevelSetField::normal));
//  VecDestroy(&(LevelSetField::curv));
//}

//std::vector<std::shared_ptr<ablate::domain::FieldDescription>> LevelSetField::GetFields() {
//    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> levelSetField{
//        std::make_shared<domain::FieldDescription>("level set field", "phi", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM, region)};

//  return levelSetField;
//}




//#include "registrar.hpp"
//REGISTER(ablate::domain::FieldDescriptor, LevelSetField, "Level Set fields need for interface tracking",
//         OPT(ablate::domain::Region, "region", "the region for the level set (defaults to entire domain)"));

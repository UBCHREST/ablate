#include "levelSetSolver.hpp"
#include "levelSetUtilities.hpp"
#include "domain/range.hpp"

using namespace ablate::levelSet;

LevelSetSolver::LevelSetSolver(std::string solverId, std::shared_ptr<ablate::domain::Region> region, std::shared_ptr<ablate::parameters::Parameters> options,
const std::shared_ptr<ablate::domain::rbf::RBF>& rbf) : Solver(solverId, region, options), rbf(rbf) {}


// This is done once
void LevelSetSolver::Setup() {

// Make sure that the level set field has been created in the YAML file.
  if (!(subDomain->ContainsField(LevelSetFields::LEVELSET_FIELD))) {
    throw std::runtime_error("ablate::levelSet::LevelSetSolver expects a level set field to be defined.");
  }
  if (!(subDomain->ContainsField(LevelSetFields::CURVATURE_FIELD))) {
    throw std::runtime_error("ablate::levelSet::LevelSetSolver expects a curvature field to be defined.");
  }
  if (!(subDomain->ContainsField(LevelSetFields::NORMAL_FIELD))) {
    throw std::runtime_error("ablate::levelSet::LevelSetSolver expects a normal field to be defined.");
  }

  // Store the fields for easy access later
  LevelSetSolver::lsField = &(subDomain->GetField(LevelSetFields::LEVELSET_FIELD));
  LevelSetSolver::curvField = &(subDomain->GetField(LevelSetFields::CURVATURE_FIELD));
  LevelSetSolver::normalField = &(subDomain->GetField(LevelSetFields::NORMAL_FIELD));

  // Setup the RBF
  LevelSetSolver::rbf->Setup(subDomain);
}

// Done whenever the subDomain changes
void LevelSetSolver::Initialize() {

  // Initialize the RBF data structures
  LevelSetSolver::rbf->Initialize();

}

///*************   Begin Curvature and Normal Vector functions ******************/

// Might want to have this based on cell values rather than RBF
void LevelSetSolver::Normal1D(PetscInt c, PetscScalar *n) {

  PetscReal cx = 0.0, g = 0.0;
  std::shared_ptr<ablate::domain::rbf::RBF> rbf = LevelSetSolver::rbf;
  const ablate::domain::Field *lsField = LevelSetSolver::lsField;

  cx = rbf->EvalDer(lsField, c, 1, 0, 0);
  g = PetscSqrtReal(cx*cx);

  n[0] = cx/g;
}

void LevelSetSolver::Normal2D(PetscInt c, PetscScalar *n) {

  PetscReal   cx = 0.0, cy = 0.0, g = 0.0;
  std::shared_ptr<ablate::domain::rbf::RBF> rbf = LevelSetSolver::rbf;
  const ablate::domain::Field *lsField = LevelSetSolver::lsField;

  cx = rbf->EvalDer(lsField, c, 1, 0, 0);
  cy = rbf->EvalDer(lsField, c, 0, 1, 0);
  g = PetscSqrtReal(cx*cx + cy*cy);

  n[0] = cx/g;
  n[1] = cy/g;


}

void LevelSetSolver::Normal3D(PetscInt c, PetscReal *n) {

  PetscReal   cx = 0.0, cy = 0.0, cz = 0.0, g = 0.0;
  std::shared_ptr<ablate::domain::rbf::RBF> rbf = LevelSetSolver::rbf;
  const ablate::domain::Field *lsField = LevelSetSolver::lsField;

  cx = rbf->EvalDer(lsField, c, 1, 0, 0);
  cy = rbf->EvalDer(lsField, c, 0, 1, 0);
  cz = rbf->EvalDer(lsField, c, 0, 0, 1);
  g = sqrt(cx*cx + cy*cy + cz*cz);

  n[0] = cx/g;
  n[1] = cy/g;
  n[2] = cz/g;
}

PetscReal LevelSetSolver::Curvature1D(PetscInt c) {
  return 0.0;
}

PetscReal LevelSetSolver::Curvature2D(PetscInt c) {

  PetscReal k = 0.0;
  PetscReal cx, cy, cxx, cyy, cxy;
  std::shared_ptr<ablate::domain::rbf::RBF> rbf = LevelSetSolver::rbf;
  const ablate::domain::Field *lsField = LevelSetSolver::lsField;

  cx = rbf->EvalDer(lsField, c, 1, 0, 0);
  cy = rbf->EvalDer(lsField, c, 0, 1, 0);
  cxx = rbf->EvalDer(lsField, c, 2, 0, 0);
  cyy = rbf->EvalDer(lsField, c, 0, 2, 0);
  cxy = rbf->EvalDer(lsField, c, 1, 1, 0);

  k = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/pow(cx*cx+cy*cy,1.5);

  return k;
}

PetscReal LevelSetSolver::Curvature3D(PetscInt c) {

  PetscReal k = 0.0;
  PetscReal cx, cy, cz;
  PetscReal cxx, cyy, czz;
  PetscReal cxy, cxz, cyz;
  std::shared_ptr<ablate::domain::rbf::RBF> rbf = LevelSetSolver::rbf;
  const ablate::domain::Field *lsField = LevelSetSolver::lsField;

  cx = rbf->EvalDer(lsField, c, 1, 0, 0);
  cy = rbf->EvalDer(lsField, c, 0, 1, 0);
  cz = rbf->EvalDer(lsField, c, 0, 0, 1);
  cxx = rbf->EvalDer(lsField, c, 2, 0, 0);
  cyy = rbf->EvalDer(lsField, c, 0, 2, 0);
  czz = rbf->EvalDer(lsField, c, 0, 0, 2);
  cxy = rbf->EvalDer(lsField, c, 1, 1, 0);
  cxz = rbf->EvalDer(lsField, c, 1, 0, 1);
  cyz = rbf->EvalDer(lsField, c, 0, 1, 1);

  k = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2.0*(cxy*cx*cy + cxz*cx*cz + cyz*cy*cz))/pow(cx*cx+cy*cy+cz*cz,1.5);

  return k;
}

// Maybe have it point directly to the function during setup
PetscReal LevelSetSolver::Curvature(PetscInt c) {
  switch (subDomain->GetDimensions()) {
    case 1:
      return 0.0;
    case 2:
      return LevelSetSolver::Curvature2D(c);
    case 3:
      return LevelSetSolver::Curvature3D(c);
    default:
      throw std::runtime_error("ablate::levelSet::LevelSetSolver::Curvature encountered an unknown dimension.");
  }
}

void LevelSetSolver::Normal(PetscInt c, PetscReal *n) {
  switch (subDomain->GetDimensions()) {
    case 1:
      throw std::runtime_error("Still need to setup ablate::levelSet::LevelSetSolver::Normal for 1D grids.\n");
    case 2:
      return LevelSetSolver::Normal2D(c, n);
    case 3:
      return LevelSetSolver::Normal3D(c, n);
    default:
      throw std::runtime_error("ablate::levelSet::LevelSetSolver::Normal encountered an unknown dimension.");
  }
}

void LevelSetSolver::ComputeAllNormal() {
  DM            dm = subDomain->GetDM();
  domain::Range cellRange;
  PetscReal    *array, *n;
  Vec           auxVec = subDomain->GetAuxVector();       // For normal vector

  GetCellRange(cellRange);
  VecGetArray(auxVec, &array) >> utilities::PetscUtilities::checkError;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.points ? cellRange.points[c] : c;
    DMPlexPointLocalFieldRef(dm, cell, LevelSetSolver::normalField->id, array, &n) >> utilities::PetscUtilities::checkError;
    LevelSetSolver::Normal(cell, n);
  }
  VecRestoreArray(auxVec, &array) >> utilities::PetscUtilities::checkError;
  RestoreRange(cellRange);
}

void LevelSetSolver::ComputeAllCurvature() {
  DM            dm = subDomain->GetDM();
  domain::Range cellRange;
  PetscReal    *array, *h;
  Vec           auxVec = subDomain->GetAuxVector();       // For normal vector

  GetCellRange(cellRange);
  VecGetArray(auxVec, &array) >> utilities::PetscUtilities::checkError;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.points ? cellRange.points[c] : c;
    DMPlexPointLocalFieldRef(dm, cell, LevelSetSolver::curvField->id, array, &h);
    h[0] = LevelSetSolver::Curvature(cell);
  }
  VecRestoreArray(auxVec, &array) >> utilities::PetscUtilities::checkError;
  RestoreRange(cellRange);
}

/*************   End Curvature and Normal Vector functions ******************/

// Returns the VOF for a given cell. Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
//  by Holdych, Noble, and Secor, Int. J. Numer. Meth. Engng 2008; 73:1310-1327.
void LevelSetSolver::VOF(const PetscInt p, PetscReal *vof, PetscReal *area, PetscReal *vol) {

  DM                dm = subDomain->GetDM();
  PetscScalar       c0, n[3] = {0.0, 0.0, 0.0};
  const PetscScalar *array;
  Vec               solVec = subDomain->GetSolutionVector();  // For level set
  Vec               auxVec = subDomain->GetAuxVector();       // For normal vector

  // Level-set value at cell-center
  VecGetArrayRead(solVec, &array) >> ablate::utilities::PetscUtilities::checkError;
  DMPlexPointLocalFieldRead(dm, p, LevelSetSolver::lsField->id, array, &c0) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(solVec, &array) >> ablate::utilities::PetscUtilities::checkError;

  // Normal vector
  VecGetArrayRead(auxVec, &array) >> ablate::utilities::PetscUtilities::checkError;
  DMPlexPointLocalFieldRead(dm, p, LevelSetSolver::normalField->id, array, n) >> ablate::utilities::PetscUtilities::checkError;
  VecRestoreArrayRead(auxVec, &array) >> ablate::utilities::PetscUtilities::checkError;

  ablate::levelSet::Utilities::VOF(dm, p, c0, n, vof, area, vol);

}


//// Given a VOF field compute the level set field and normal vectors at the cell center
//void LevelSetSolver::Reinitialize(const ablate::domain::Field *field, Vec f) {

//  DM vertexDM;

//  DMClone(subDomain->GetDM(), &vertexDM);



//  ablate::domain::Range vertexRange;

//  // This is from the solver
////  ablate::solver::Solver::GetRange((0, &vertexRange);
//  GetCellRange(&vertexRange);





//}







//// Reinitialize a level set field to make it a signed distance function and to match a target VOF for each cell
//void LevelSetSolver::Reinitialize(TS ts, ablate::solver::Solver &solver) {
////    // Get the solution vec and dm
////    auto dm = solver.GetSubDomain().GetDM();
////    auto solVec = solver.GetSubDomain().GetSolutionVector();
////    auto auxDm = solver.GetSubDomain().GetAuxDM();
////    auto auxVec = solver.GetSubDomain().GetAuxVector();

////    // Get the array vector
////    PetscScalar *solutionArray;
////    VecGetArray(solVec, &solutionArray) >> utilities::PetscUtilities::checkError;
////    PetscScalar *auxArray;
////    VecGetArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;

////    // March over each cell in this domain
////    solver::Range cellRange;
////    solver.GetCellRange(cellRange);
////    auto dim = solver.GetSubDomain().GetDimensions();

////    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
////        PetscInt cell = cellRange.points ? cellRange.points[c] : c;


////        // Get the euler and density field
////        const PetscScalar *euler = nullptr;
////        DMPlexPointGlobalFieldRef(dm, cell, eulerFieldInfo.id, solutionArray, &euler) >> utilities::PetscUtilities::checkError;
////        PetscScalar *densityYi;
////        DMPlexPointGlobalFieldRef(dm, cell, densityYiFieldInfo.id, solutionArray, &densityYi) >> utilities::PetscUtilities::checkError;
////        PetscScalar *yi;
////        DMPlexPointLocalFieldRead(auxDm, cell, yiFieldInfo.id, auxArray, &yi) >> utilities::PetscUtilities::checkError;
////        PetscFVCellGeom *cellGeom;
////        DMPlexPointLocalRead(cellGeomDm, cell, cellGeomArray, &cellGeom) >> utilities::PetscUtilities::checkError;

////        // compute the mass fractions on the boundary
////        massFractionsFunction(dim, time, cellGeom->centroid, yiFieldInfo.numberComponents, yi, massFractionsContext);

////        // Only update if in the global vector
////        if (euler) {
////            // Get density
////            const PetscScalar density = euler[finiteVolume::CompressibleFlowFields::RHO];

////            for (PetscInt sp = 0; sp < densityYiFieldInfo.numberComponents; sp++) {
////                densityYi[sp] = yi[sp] * density;
////            }
////        }
////    }

////    // cleanup
////    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
////    VecRestoreArray(auxVec, &auxArray) >> utilities::PetscUtilities::checkError;
////    VecRestoreArray(solVec, &solutionArray) >> utilities::PetscUtilities::checkError;
////    solver.RestoreRange(cellRange);









////  PetscInt          c, cStart, cEnd;
////  DM                dm = LevelSetField::dm;
////  const PetscScalar *vofVal;
////  PetscScalar       *phiVal;
////  PetscReal         vof, faceArea, cellVolume;
////  Vec               newPhi;

////  VecDuplicate(LevelSetField::phi, &newPhi);

////  VecDuplicate(newPhi, &newPhi);

////  VecGetArrayRead(VOF, &vofVal) >> utilities::PetscUtilities::checkError;
////  VecGetArray(newPhi, &phiVal) >> utilities::PetscUtilities::checkError;

//////Take a look at boundarySolver/physics/sublimation.cpp lines 233-246 + 276

//////Use DMPlexPointGlobalFieldRead to get field values
//////Stuff like const auto &eulerFieldInfo = solver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD); will return the field info in the DM.
//////Make the level set a solution variable in the ablate solver


////  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
////  for (c = cStart; c < cEnd; ++c) {
////    LevelSetField::VOF(c, &vof, &faceArea, &cellVolume);

////  }

////  VecRestoreArray(newPhi, &phiVal);
////  VecRestoreArrayRead(VOF, &vofVal) >> utilities::PetscUtilities::checkError;
////  VecDestroy(&newPhi);


//}


////bool LevelSetField::HasInterface(const PetscInt p) {
////  bool              hasInterface = false;
////  PetscInt          nCells = 0, *cells = NULL;
////  PetscInt          i, cStart;
////  Vec               phi = LevelSetField::phi;
////  const PetscScalar *array;
////  PetscScalar       c0;
////  DM                dm = LevelSetField::dm;

////  DMPlexGetHeightStratum(dm, 0, &cStart, NULL) >> utilities::PetscUtilities::checkError;

////  DMPlexGetNeighborCells(dm, p, 1, -1, -1, PETSC_TRUE, &nCells, &cells) >> utilities::PetscUtilities::checkError;

////  VecGetArrayRead(phi, &array) >> utilities::PetscUtilities::checkError;
////  c0 = array[p - cStart];

////  i = 0;
////  while (i < nCells && !hasInterface) {
////    hasInterface = ((c0 * array[cells[i] - cStart])<=0.0);
////    ++i;
////  }

////  VecRestoreArrayRead(phi, &array) >> utilities::PetscUtilities::checkError;
////  PetscFree(cells) >> utilities::PetscUtilities::checkError;

////  return hasInterface;

////}

/////* Sphere */
////PetscReal LevelSetField::Sphere(PetscReal pos[], PetscReal center[], PetscReal radius) {
////  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
////  PetscReal phi = PetscSqrtReal(PetscSqr(shiftedPos[0]) + PetscSqr(shiftedPos[1]) + PetscSqr(shiftedPos[2])) - radius;
////  return phi;
////}

/////* Ellipse */
////PetscReal LevelSetField::Ellipse(PetscReal pos[], PetscReal center[], PetscReal radius) {
////  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
////  PetscReal phi = PetscSqr(shiftedPos[0]/0.5) + PetscSqr(shiftedPos[1]/1.25) + PetscSqr(shiftedPos[2]) - radius;
////  return phi;
////}


/////* Star */
////PetscReal LevelSetField::Star(PetscReal pos[], PetscReal center[]) {
////  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
////  PetscReal phi = 400.0*shiftedPos[0]*shiftedPos[0]*shiftedPos[1]*shiftedPos[1]-(1.0-0.5*shiftedPos[0]*shiftedPos[0]-0.5*shiftedPos[1]*shiftedPos[1]);
////  return phi;
////}

//PetscReal LevelSetSolver::Interpolate(PetscScalar xyz[3]) {
//  std::shared_ptr<ablate::domain::rbf::RBF>  rbf = LevelSetSolver::rbf;
//  DMInterpolationInfo   ctx;
//  DM                    dm = rbf->GetDM();
//  PetscInt              c = -1;
//  Vec                   phi = LevelSetField::phi;
//  PetscReal             val;

//  DMInterpolationCreate(PETSC_COMM_WORLD, &ctx) >> utilities::PetscUtilities::checkError;
//  DMInterpolationSetDim(ctx, LevelSetField::dim) >> utilities::PetscUtilities::checkError;
//  DMInterpolationAddPoints(ctx, 1, xyz) >> utilities::PetscUtilities::checkError;
//  DMInterpolationSetUp(ctx, dm, PETSC_FALSE, PETSC_FALSE) >> utilities::PetscUtilities::checkError;
//  c = ctx->cells[0];
//  DMInterpolationDestroy(&ctx) >> utilities::PetscUtilities::checkError;


//  PetscReal RBF::Interpolate(const ablate::domain::Field *field, PetscInt c, PetscReal xEval[3]) {

//  val = rbf->Interpolate(phi, c, xyz);

//  return val;
//}

//PetscReal LevelSetSolver::Interpolate(const PetscReal x, const double y, const double z) {

//  PetscReal xyz[3] = {x, y, z};
//  PetscReal val = LevelSetSolver::Interpolate(xyz);

//  return val;
//}



////void LevelSetField::Advect(Vec velocity, const PetscReal dt) {

////  Vec               phi = LevelSetField::phi, nextPhi = nullptr;
////  DM                dm = LevelSetField::dm;
////  PetscInt          dim = LevelSetField::dim;
////  PetscInt          cStart, cEnd, c, cShift;
////  PetscScalar       *newVal;
////  const PetscScalar *vel;
////  PetscReal         pos[3] = {0.0, 0.0, 0.0};


////  VecDuplicate(phi, &nextPhi);

////  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;       // Range of cells

////  VecGetArray(nextPhi, &newVal) >> utilities::PetscUtilities::checkError;
////  VecGetArrayRead(velocity, &vel) >> utilities::PetscUtilities::checkError;
////  for (c = cStart; c < cEnd; ++c) {
////    cShift = c - cStart;
////    // Cell center
////    DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> utilities::PetscUtilities::checkError;

////    // Step backward
////    for (PetscInt d = 0; d < dim; ++d) {
////      pos[d] -= dt*vel[cShift*dim + d];
////    }

////    newVal[cShift] = LevelSetField::Interpolate(pos);
////  }
////  VecRestoreArrayRead(velocity, &vel) >> utilities::PetscUtilities::checkError;
////  VecRestoreArray(nextPhi, &newVal) >> utilities::PetscUtilities::checkError;

////  VecCopy(nextPhi, phi) >> utilities::PetscUtilities::checkError;
////  VecDestroy(&nextPhi) >> utilities::PetscUtilities::checkError;

////  VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD) >> utilities::PetscUtilities::checkError;
////  VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD) >> utilities::PetscUtilities::checkError;


////  VecDestroy(&nextPhi);



////}



//// Reinitialize a level set field to make it a signed distance function and to match a target VOF for each cell
//void LevelSetField::Reinitialize(Vec VOF) {
//  PetscInt          c, cStart, cEnd;
//  DM                dm = LevelSetField::dm;
//  const PetscScalar *vofVal;
//  PetscScalar       *phiVal;
//  PetscReal         vof, faceArea, cellVolume;
//  Vec               newPhi;

//  VecDuplicate(LevelSetField::phi, &newPhi);

//  VecDuplicate(newPhi, &newPhi);

//  VecGetArrayRead(VOF, &vofVal) >> utilities::PetscUtilities::checkError;
//  VecGetArray(newPhi, &phiVal) >> utilities::PetscUtilities::checkError;

////Take a look at boundarySolver/physics/sublimation.cpp lines 233-246 + 276

////Use DMPlexPointGlobalFieldRead to get field values
////Stuff like const auto &eulerFieldInfo = solver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD); will return the field info in the DM.
////Make the level set a solution variable in the ablate solver


//  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
//  for (c = cStart; c < cEnd; ++c) {
//    LevelSetField::VOF(c, &vof, &faceArea, &cellVolume);

//  }

//  VecRestoreArray(newPhi, &phiVal);
//  VecRestoreArrayRead(VOF, &vofVal) >> utilities::PetscUtilities::checkError;
//  VecDestroy(&newPhi);


//}


#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::levelSet::LevelSetSolver, "level set solver",
         ARG(std::string, "id", "the name of the level set solver"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the solver"),
         OPT(ablate::domain::rbf::RBF, "rbf", "The radial basis function to use"));

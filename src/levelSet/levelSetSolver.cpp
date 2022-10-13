#include "levelSetSolver.hpp"
//#include <petsc/private/dmpleximpl.h>
//#include "utilities/mpiError.hpp"
//#include "utilities/petscError.hpp"

using namespace ablate::levelSet;



//ablate::levelSet::LevelSetSolver::LevelSetSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
//    : Solver(std::move(solverId), std::move(region), std::move(options)) {}

LevelSetSolver::LevelSetSolver(std::string solverId, std::shared_ptr<ablate::domain::Region> region, std::shared_ptr<ablate::parameters::Parameters> options) : Solver(solverId, region, options) { }


// This is done once
void LevelSetSolver::Setup() {


  // Make sure that the level set field has been created in the YAML file.
  if (!(subDomain->ContainsField(LevelSetSolver::LEVELSET_FIELD))) {
    throw std::runtime_error("ablate::levelSet::LevelSetSolver expects a level set field to be defined.");
  }

  // Make sure that the RBF is setup.
  subDomain->SetupRBF(subDomain);

  LevelSetSolver::lsField = &(subDomain->GetField(LevelSetSolver::LEVELSET_FIELD));

  // Save the dimension
  LevelSetSolver::dim = subDomain->GetDimensions();



printf("All Done!\n");
PetscFinalize();


}

void LevelSetSolver::Initialize() {

}

/*************   Begin Curvature and Normal Vector functions ******************/

void LevelSetSolver::Normal2D(PetscInt c, PetscScalar *n) {

  PetscReal   cx = 0.0, cy = 0.0, g = 0.0;
  std::shared_ptr<ablate::domain::RBF> rbf = subDomain->GetRBF();

  cx = rbf->EvalDer(LevelSetSolver::lsField, c, 1, 0, 0);
  cy = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 1, 0);
  g = PetscSqrtReal(cx*cx + cy*cy);

  n[0] = cx/g;
  n[1] = cy/g;


}

void LevelSetSolver::Normal3D(PetscInt c, PetscReal *n) {

  PetscReal   cx = 0.0, cy = 0.0, cz = 0.0, g = 0.0;
  std::shared_ptr<ablate::domain::RBF> rbf = subDomain->GetRBF();

  cx = rbf->EvalDer(LevelSetSolver::lsField, c, 1, 0, 0);
  cy = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 1, 0);
  cz = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 0, 1);
  g = sqrt(cx*cx + cy*cy + cz*cz);

  n[0] = cx/g;
  n[1] = cy/g;
  n[2] = cz/g;
}

PetscReal LevelSetSolver::Curvature2D(PetscInt c) {

  PetscReal k = 0.0;
  PetscReal cx, cy, cxx, cyy, cxy;
  std::shared_ptr<ablate::domain::RBF> rbf = subDomain->GetRBF();

  cx = rbf->EvalDer(LevelSetSolver::lsField, c, 1, 0, 0);
  cy = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 1, 0);
  cxx = rbf->EvalDer(LevelSetSolver::lsField, c, 2, 0, 0);
  cyy = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 2, 0);
  cxy = rbf->EvalDer(LevelSetSolver::lsField, c, 1, 1, 0);

  k = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/pow(cx*cx+cy*cy,1.5);

  return k;
}

PetscReal LevelSetSolver::Curvature3D(PetscInt c) {

  PetscReal k = 0.0;
  PetscReal cx, cy, cz;
  PetscReal cxx, cyy, czz;
  PetscReal cxy, cxz, cyz;
  std::shared_ptr<ablate::domain::RBF> rbf = subDomain->GetRBF();

  cx = rbf->EvalDer(LevelSetSolver::lsField, c, 1, 0, 0);
  cy = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 1, 0);
  cz = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 0, 1);
  cxx = rbf->EvalDer(LevelSetSolver::lsField, c, 2, 0, 0);
  cyy = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 2, 0);
  czz = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 0, 2);
  cxy = rbf->EvalDer(LevelSetSolver::lsField, c, 1, 1, 0);
  cxz = rbf->EvalDer(LevelSetSolver::lsField, c, 1, 0, 1);
  cyz = rbf->EvalDer(LevelSetSolver::lsField, c, 0, 1, 1);

  k = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2.0*(cxy*cx*cy + cxz*cx*cz + cyz*cy*cz))/pow(cx*cx+cy*cy+cz*cz,1.5);

  return k;
}

// There has to be a better way of doing this so that the curvature/normal function points directly to either 2D or 3D during setup.
PetscReal LevelSetSolver::Curvature(PetscInt c) {
  if (LevelSetSolver::dim==2) {
    return LevelSetSolver::Curvature2D(c);
  }
  else {
    return LevelSetSolver::Curvature3D(c);
  }
}

void LevelSetSolver::Normal(PetscInt c, PetscReal *n) {
  if (LevelSetSolver::dim==2) {
    return LevelSetSolver::Normal2D(c, n);
  }
  else {
    return LevelSetSolver::Normal3D(c, n);
  }
}

/*************   End Curvature and Normal Vector functions ******************/


// Reinitialize a level set field to make it a signed distance function and to match a target VOF for each cell
void LevelSetSolver::Reinitialize(TS ts, ablate::solver::Solver &solver) {
//    // Get the solution vec and dm
//    auto dm = solver.GetSubDomain().GetDM();
//    auto solVec = solver.GetSubDomain().GetSolutionVector();
//    auto auxDm = solver.GetSubDomain().GetAuxDM();
//    auto auxVec = solver.GetSubDomain().GetAuxVector();

//    // Get the array vector
//    PetscScalar *solutionArray;
//    VecGetArray(solVec, &solutionArray) >> checkError;
//    PetscScalar *auxArray;
//    VecGetArray(auxVec, &auxArray) >> checkError;

//    // March over each cell in this domain
//    solver::Range cellRange;
//    solver.GetCellRange(cellRange);
//    auto dim = solver.GetSubDomain().GetDimensions();

//    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//        PetscInt cell = cellRange.points ? cellRange.points[c] : c;


//        // Get the euler and density field
//        const PetscScalar *euler = nullptr;
//        DMPlexPointGlobalFieldRef(dm, cell, eulerFieldInfo.id, solutionArray, &euler) >> checkError;
//        PetscScalar *densityYi;
//        DMPlexPointGlobalFieldRef(dm, cell, densityYiFieldInfo.id, solutionArray, &densityYi) >> checkError;
//        PetscScalar *yi;
//        DMPlexPointLocalFieldRead(auxDm, cell, yiFieldInfo.id, auxArray, &yi) >> checkError;
//        PetscFVCellGeom *cellGeom;
//        DMPlexPointLocalRead(cellGeomDm, cell, cellGeomArray, &cellGeom) >> checkError;

//        // compute the mass fractions on the boundary
//        massFractionsFunction(dim, time, cellGeom->centroid, yiFieldInfo.numberComponents, yi, massFractionsContext);

//        // Only update if in the global vector
//        if (euler) {
//            // Get density
//            const PetscScalar density = euler[finiteVolume::CompressibleFlowFields::RHO];

//            for (PetscInt sp = 0; sp < densityYiFieldInfo.numberComponents; sp++) {
//                densityYi[sp] = yi[sp] * density;
//            }
//        }
//    }

//    // cleanup
//    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
//    VecRestoreArray(auxVec, &auxArray) >> checkError;
//    VecRestoreArray(solVec, &solutionArray) >> checkError;
//    solver.RestoreRange(cellRange);









//  PetscInt          c, cStart, cEnd;
//  DM                dm = LevelSetField::dm;
//  const PetscScalar *vofVal;
//  PetscScalar       *phiVal;
//  PetscReal         vof, faceArea, cellVolume;
//  Vec               newPhi;

//  VecDuplicate(LevelSetField::phi, &newPhi);

//  VecDuplicate(newPhi, &newPhi);

//  VecGetArrayRead(VOF, &vofVal) >> ablate::checkError;
//  VecGetArray(newPhi, &phiVal) >> ablate::checkError;

////Take a look at boundarySolver/physics/sublimation.cpp lines 233-246 + 276

////Use DMPlexPointGlobalFieldRead to get field values
////Stuff like const auto &eulerFieldInfo = solver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD); will return the field info in the DM.
////Make the level set a solution variable in the ablate solver


//  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::checkError;
//  for (c = cStart; c < cEnd; ++c) {
//    LevelSetField::VOF(c, &vof, &faceArea, &cellVolume);

//  }

//  VecRestoreArray(newPhi, &phiVal);
//  VecRestoreArrayRead(VOF, &vofVal) >> ablate::checkError;
//  VecDestroy(&newPhi);


}


//std::string LevelSetSolver::GetRBFType() {

//  switch (LevelSetSolver::rbfType) {
//    case ablate::RBF::RBFType::PHS:
//      return("phs");
//    case ablate::RBF::RBFType::MQ:
//      return("mq");
//    case ablate::RBF::RBFType::IMQ:
//      return("imq");
//    case ablate::RBF::RBFType::GA:
//      return("ga");
//    default:
//      return("unknown");
//  }
//}



#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::levelSet::LevelSetSolver, "level set solver",
         ARG(std::string, "id", "the name of the level set"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"));


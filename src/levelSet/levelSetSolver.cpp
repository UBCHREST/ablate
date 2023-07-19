#include "domain/range.hpp"
#include "levelSetSolver.hpp"
#include "levelSetUtilities.hpp"
#include "geometry.hpp"

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

//void LevelSetSolver::ComputeAllNormal() {
//  DM            dm = subDomain->GetDM();
//  domain::Range cellRange;
//  PetscReal    *array, *n;
//  Vec           auxVec = subDomain->GetAuxVector();       // For normal vector

//  GetCellRange(cellRange);
//  VecGetArray(auxVec, &array) >> utilities::PetscUtilities::checkError;
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    PetscInt cell = cellRange.points ? cellRange.points[c] : c;
//    DMPlexPointLocalFieldRef(dm, cell, LevelSetSolver::normalField->id, array, &n) >> utilities::PetscUtilities::checkError;
//    ablate::levelSet::geometry::Normal(rbf, lsField, cell, n);
//  }
//  VecRestoreArray(auxVec, &array) >> utilities::PetscUtilities::checkError;
//  RestoreRange(cellRange);
//}

//void LevelSetSolver::ComputeAllCurvature() {
//  DM            dm = subDomain->GetDM();
//  domain::Range cellRange;
//  PetscReal    *array, *h;
//  Vec           auxVec = subDomain->GetAuxVector();       // For normal vector

//  GetCellRange(cellRange);
//  VecGetArray(auxVec, &array) >> utilities::PetscUtilities::checkError;
//  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
//    PetscInt cell = cellRange.points ? cellRange.points[c] : c;
//    DMPlexPointLocalFieldRef(dm, cell, LevelSetSolver::curvField->id, array, &h);
//    *h = ablate::levelSet::geometry::Curvature(rbf, lsField, cell);
//  }
//  VecRestoreArray(auxVec, &array) >> utilities::PetscUtilities::checkError;
//  RestoreRange(cellRange);
//}

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

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::levelSet::LevelSetSolver, "level set solver",
         ARG(std::string, "id", "the name of the level set solver"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the solver"),
         OPT(ablate::domain::rbf::RBF, "rbf", "The radial basis function to use"));

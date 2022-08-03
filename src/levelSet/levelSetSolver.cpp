#include "levelSetSolver.hpp"
//#include <petsc/private/dmpleximpl.h>
//#include "utilities/mpiError.hpp"
//#include "utilities/petscError.hpp"

using namespace ablate::levelSet;


//ablate::levelSet::LevelSetSolver::LevelSetSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
//    : Solver(std::move(solverId), std::move(region), std::move(options)) {}

LevelSetSolver::LevelSetSolver(std::shared_ptr<ablate::levelSet::LevelSetField> lsField) {
  LevelSetSolver::lsField = lsField;
}

//void ablate::levelSet::LevelSetSolver::Setup() {
//  DM cdm = subDomain->GetDM();

//  while (cdm) {
//      DMCopyDisc(subDomain->GetDM(), cdm) >> checkError;
//      DMGetCoarseDM(cdm, &cdm) >> checkError;
//  }

//  //// Register the aux fields updater if specified
//  //if (!auxiliaryFieldsUpdaters.empty()) {
//      //RegisterPreStep([&](TS ts, Solver &) { UpdateAuxFields(ts, *this); });
//  //}

//  //// add each boundary condition
//  //for (auto boundary : boundaryConditions) {
//      //const auto &fieldId = subDomain->GetField(boundary->GetFieldName());

//      //// Setup the boundary condition
//      //boundary->SetupBoundary(subDomain->GetDM(), subDomain->GetDiscreteSystem(), fieldId.id);
//  //}
//}

//void ablate::levelSet::LevelSetSolver::Initialize() {

//}



//#include "registrar.hpp"
//REGISTER(ablate::solver::Solver, ablate::levelSet::LevelSetSolver, "level set solver", ARG(std::string, "id", "the name of the level set"),
//         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
//         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"));

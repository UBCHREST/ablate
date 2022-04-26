#ifndef ABLATELIBRARY_LEVELSETSOLVER_HPP
#define ABLATELIBRARY_LEVELSETSOLVER_HPP

#include <petsc.h>
#include <string>
#include <vector>
//#include "boundaryConditions/boundaryCondition.hpp"
//#include "mathFunctions/fieldFunction.hpp"
//#include "solver/cellSolver.hpp"
#include "solver/solver.hpp"
//#include "solver/timeStepper.hpp"


namespace ablate::levelSet {

class LevelSetSolver : public solver::Solver{
  private:


  public:
    LevelSetSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options);

    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;

};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_LEVELSETSOLVER_HPP

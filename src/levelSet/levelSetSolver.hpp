#ifndef ABLATELIBRARY_LEVELSETSOLVER_HPP
#define ABLATELIBRARY_LEVELSETSOLVER_HPP

#include <petsc.h>
#include <string>
#include <vector>
//#include "boundaryConditions/boundaryCondition.hpp"
//#include "mathFunctions/fieldFunction.hpp"
//#include "solver/cellSolver.hpp"
#include "solver/solver.hpp"
#include "levelSetField.hpp"
#include "solver/timeStepper.hpp"

//ablate::solver::Solver::Solver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)

class LevelSetSolver {
  private:
    // The level set
//    std::shared_ptr<LevelSetField> lsField = nullptr;

//    LevelSetField lsField;
//    auto lsField = std::make_shared<ablate::levelSet::LevelSetField>

  public:
//    LevelSetSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options);
    // Constructor
//    LevelSetSolver(std::shared_ptr<LevelSetField> lsField = nullptr);
    LevelSetSolver(
      std::string solverId,
      std::shared_ptr<ablate::domain::Region>,
      std::shared_ptr<ablate::parameters::Parameters> options);


//    /** SubDomain Register and Setup **/
//    void Setup() override;
//    void Initialize() override;

};

#endif  // ABLATELIBRARY_LEVELSETSOLVER_HPP

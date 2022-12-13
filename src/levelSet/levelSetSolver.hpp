#ifndef ABLATELIBRARY_LEVELSETSOLVER_HPP
#define ABLATELIBRARY_LEVELSETSOLVER_HPP

#include <petsc.h>
#include <string>
#include <vector>
//#include "mathFunctions/fieldFunction.hpp"
#include "domain/domain.hpp"
#include "solver/solver.hpp"
#include "solver/timeStepper.hpp"
#include "domain/RBF/rbf.hpp"

//ablate::solver::Solver::Solver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)

namespace ablate::levelSet {


class LevelSetSolver : public ablate::solver::Solver {
  private:
//    const ablate::RBF::RBFType rbfType = ablate::RBF::RBFType::MQ;

    void Reinitialize(TS ts, ablate::solver::Solver &solver);

    PetscInt dim;


    // Internal curvature and normal calculations
    PetscReal Curvature2D(PetscInt c);
    PetscReal Curvature3D(PetscInt c);
    void Normal2D(PetscInt c, PetscReal *n);
    void Normal3D(PetscInt c, PetscReal *n);


    const ablate::domain::Field *lsField = NULL;

  public:

    inline const static std::string LEVELSET_FIELD = "levelSet";

    LevelSetSolver(
      std::string solverId,
      std::shared_ptr<ablate::domain::Region>,
      std::shared_ptr<ablate::parameters::Parameters> options);


    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;


    // Public curvature and normal functions
    PetscReal Curvature(PetscInt c);
    void Normal(PetscInt c, PetscReal *n);

//    std::string GetRBFType();


};

}

#endif  // ABLATELIBRARY_LEVELSETSOLVER_HPP

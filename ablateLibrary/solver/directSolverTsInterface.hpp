#ifndef ABLATELIBRARY_DIRECTSOLVERTSINTERFACE_HPP
#define ABLATELIBRARY_DIRECTSOLVERTSINTERFACE_HPP
#include <petsc.h>
#include "solver.hpp"

namespace ablate::solver {

/** This static class helps setup a single solver to be used by a ts directly **/
class DirectSolverTsInterface {
   private:
    const std::vector<std::shared_ptr<Solver>> solvers;

    static PetscErrorCode PreStage(TS ts, PetscReal stagetime);
    static PetscErrorCode PreStep(TS ts);
    static PetscErrorCode PostStep(TS ts);
    static PetscErrorCode PostEvaluate(TS ts);

   public:
    DirectSolverTsInterface(TS ts, std::vector<std::shared_ptr<Solver>> solvers);
    DirectSolverTsInterface(TS ts, std::shared_ptr<Solver> solver);
};

}  // namespace ablate::solver
#endif  // ABLATELIBRARY_DIRECTSOLVERTSINTERFACE_HPP

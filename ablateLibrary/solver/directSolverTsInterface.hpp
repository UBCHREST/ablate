#ifndef ABLATELIBRARY_DIRECTSOLVERTSINTERFACE_HPP
#define ABLATELIBRARY_DIRECTSOLVERTSINTERFACE_HPP
#include <petsc.h>
#include "solver.hpp"

namespace ablate::solver {

/** This static class helps setup a single solver to be used by a ts directly **/
class DirectSolverTsInterface {
   private:
    DirectSolverTsInterface() = delete;

    static PetscErrorCode PreStage(TS ts, PetscReal stagetime);
    static PetscErrorCode PreStep(TS ts);
    static PetscErrorCode PostStep(TS ts);
    static PetscErrorCode PostEvaluate(TS ts);

   public:
    static PetscErrorCode SetupSolverTS(std::shared_ptr<Solver> solver, TS ts);
};

}  // namespace ablate::solver
#endif  // ABLATELIBRARY_DIRECTSOLVERTSINTERFACE_HPP

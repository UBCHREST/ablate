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
    static PetscErrorCode ComputeIFunction(DM, PetscReal time, Vec locX, Vec locX_t, Vec locF, void *ctx);
    static PetscErrorCode ComputeIJacobian(DM, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *ctx);
    static PetscErrorCode ComputeBoundary(DM, PetscReal time, Vec locX, Vec locX_t, void *ctx);
    static PetscErrorCode ComputeRHSFunction(DM dm, PetscReal time, Vec locX, Vec F, void *ctx);

    /** helper function to see if a class implements an interface **/
    template <class T>
    static bool AnyOfType(std::vector<std::shared_ptr<Solver>> solvers) {
        for (auto &solver : solvers) {
            if (std::dynamic_pointer_cast<T>(solver)) {
                return true;
            }
        }

        return false;
    }

   public:
    DirectSolverTsInterface(TS ts, std::vector<std::shared_ptr<Solver>> solvers);
    DirectSolverTsInterface(TS ts, std::shared_ptr<Solver> solver);
};

}  // namespace ablate::solver
#endif  // ABLATELIBRARY_DIRECTSOLVERTSINTERFACE_HPP

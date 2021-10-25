#ifndef ABLATELIBRARY_MONITOR_HPP
#define ABLATELIBRARY_MONITOR_HPP
#include <petsc.h>
#include <memory>
#include "solver/solver.hpp"

namespace ablate::monitors {

typedef PetscErrorCode (*PetscMonitorFunction)(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

class Monitor {
   private:
    std::shared_ptr<solver::Solver> solver;

   public:
    virtual ~Monitor() = default;
    virtual void* GetContext() { return this; }
    virtual void Register(std::shared_ptr<solver::Solver> solverIn) { solver = solverIn; }
    virtual PetscMonitorFunction GetPetscFunction() = 0;

   protected:
    std::shared_ptr<solver::Solver> GetSolver() { return solver; }
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_MONITOR_HPP

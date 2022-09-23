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
    /**
     * Override this function to setup the monitor
     * @param solverIn
     */
    virtual void Register(std::shared_ptr<solver::Solver> solverIn) { solver = solverIn; }

    /**
     * Return a function to be called after every time step.  By default null is returned so this is never called
     * @return
     */
    virtual PetscMonitorFunction GetPetscFunction() { return nullptr; }

    /**
     * return context to be returned to the PetscMonitorFunction.  By default this is a pointer to this instance
     */
    virtual void* GetContext() { return this; }

   protected:
    std::shared_ptr<solver::Solver> GetSolver() { return solver; }
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_MONITOR_HPP

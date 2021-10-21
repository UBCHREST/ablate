#ifndef ABLATELIBRARY_MONITOR_HPP
#define ABLATELIBRARY_MONITOR_HPP
#include <petsc.h>
#include <memory>
#include "solver/solver.hpp"

namespace ablate::monitors {

typedef PetscErrorCode (*PetscMonitorFunction)(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

class Monitor {
   public:
    virtual ~Monitor() = default;
    virtual void* GetContext() { return this; }
    virtual void Register(std::shared_ptr<solver::Solver> solver) = 0;
    virtual PetscMonitorFunction GetPetscFunction() = 0;
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_MONITOR_HPP

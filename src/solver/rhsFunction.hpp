#ifndef ABLATELIBRARY_RHSFUNCTION_HPP
#define ABLATELIBRARY_RHSFUNCTION_HPP
#include <petsc.h>

namespace ablate::solver {

class RHSFunction {
   public:
    /**
     * Called to compute the RHS source term.
     * @param time
     * @param locX The locX vector includes boundary conditions
     * @param F
     * @return
     */
    virtual PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locX, Vec F) = 0;

    /**
     * Called before the RHS function for all solvers
     * @param time
     * @param locX
     * @return
     */
    virtual PetscErrorCode PreRHSFunction(TS ts, PetscReal time, bool initialStage, Vec locX) { return 0; };
};

}  // namespace ablate::solver
#endif  // ABLATELIBRARY_RHSFUNCTION_HPP

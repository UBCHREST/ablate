#ifndef ABLATELIBRARY_RHSFUNCTION_HPP
#define ABLATELIBRARY_RHSFUNCTION_HPP
#include <petsc.h>

namespace ablate::solver {

class RHSFunction {
   public:
    virtual PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locX, Vec F) =0;
};

}
#endif  // ABLATELIBRARY_RHSFUNCTION_HPP

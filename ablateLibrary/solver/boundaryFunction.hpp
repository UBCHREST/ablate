
#ifndef ABLATELIBRARY_BOUNDARYFUNCTION_HPP
#define ABLATELIBRARY_BOUNDARYFUNCTION_HPP

#include <petsc.h>

namespace ablate::solver {

class BoundaryFunction {
   public:
    virtual PetscErrorCode ComputeBoundary(PetscReal time, Vec locX, Vec locX_t) = 0;
};

}
#endif  // ABLATELIBRARY_BOUNDARYFUNCTION_HPP

#ifndef ABLATELIBRARY_IFUNCTION_HPP
#define ABLATELIBRARY_IFUNCTION_HPP
#include <petsc.h>
namespace ablate::solver {

class IFunction {
   public:
    virtual PetscErrorCode ComputeIFunction(PetscReal time, Vec locX, Vec locX_t, Vec locF) = 0;
    virtual PetscErrorCode ComputeIJacobian(PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP) = 0;
};

}  // namespace ablate::solver
#endif  // ABLATELIBRARY_IFUNCTION_HPP

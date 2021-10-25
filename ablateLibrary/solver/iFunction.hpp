#ifndef ABLATELIBRARY_IFUNCTION_HPP
#define ABLATELIBRARY_IFUNCTION_HPP
#include <petsc.h>
namespace ablate::solver {

class IFunction {
   private:
    static PetscErrorCode StaticComputeIFunction(DM dm, PetscReal time, Vec locX, Vec locX_t, Vec locF, void* );
    static PetscErrorCode StaticComputeIJacobian(DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void* );

   public:
    virtual PetscErrorCode ComputeIFunction(PetscReal time, Vec locX, Vec locX_t, Vec locF) = 0;
    virtual PetscErrorCode ComputeIJacobian(PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP) = 0;

    /**
     * Support call to be used when using the solver directly with a TS
     */
    PetscErrorCode DMTSStaticInitialize(DM dm);
};

}    // namespace ablate::solver
#endif  // ABLATELIBRARY_IFUNCTION_HPP


#include "rhsFunction.hpp"
#include "iFunction.hpp"

PetscErrorCode ablate::solver::IFunction::StaticComputeIFunction(DM, PetscReal time, Vec locX, Vec locX_t, Vec locF, void * ctx) {
    PetscFunctionBeginUser;
    auto iFunctionSolver = (ablate::solver::IFunction*) ctx;
    PetscErrorCode ierr = iFunctionSolver->ComputeIFunction(time, locX, locX_t, locF);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::IFunction::StaticComputeIJacobian(DM, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *ctx) {
    PetscFunctionBeginUser;
    auto iFunctionSolver = (ablate::solver::IFunction*) ctx;
    PetscErrorCode ierr = iFunctionSolver->ComputeIJacobian(time, locX, locX_t, X_tShift, Jac, JacP);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::IFunction::DMTSStaticInitialize(DM dm) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ierr = DMTSSetIFunctionLocal(dm, solver::IFunction::StaticComputeIFunction, this);
    CHKERRQ(ierr);

    ierr = DMTSSetIJacobianLocal(dm, solver::IFunction::StaticComputeIJacobian, this);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);

}

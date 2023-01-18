#include "adaptPhysics.hpp"
PetscErrorCode ablate::solver::AdaptPhysics::TSAdaptChoose(TSAdapt adapt, TS ts, PetscReal h, PetscInt *next_sc, PetscReal *next_h, PetscBool *accept, PetscReal *wlte, PetscReal *wltea,
                                                           PetscReal *wlter) {
    PetscFunctionBeginUser;

    PetscCheck(adapt->always_accept, PetscObjectComm((PetscObject)adapt), PETSC_ERR_SUP, "Step rejection not implemented. The CFL implementation is incomplete/unusable");


    PetscFunctionReturn(0);
}

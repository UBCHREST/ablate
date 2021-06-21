#include "errorMonitor.hpp"
#include "mathFunctions/mathFunction.hpp"

PetscErrorCode ablate::monitors::ErrorMonitor::MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    DM dm;
    PetscDS ds;
    Vec v;

    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    // Get the number of fields
    PetscInt numberOfFields;
    ierr = PetscDSGetNumFields(ds, &numberOfFields);
    CHKERRQ(ierr);

    // Get the exact funcs and contx
    std::vector<ablate::mathFunctions::PetscFunction> exactFuncs(numberOfFields);
    std::vector<void *> exactCtxs(numberOfFields);
    for (auto f = 0; f < numberOfFields; ++f) {
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &exactCtxs[f]);
        CHKERRQ(ierr);
        if (!exactFuncs[f]) {
            SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_LIB, "The exact solution has not set");
        }
    }

    // Create an vector to hold the exact solution
    Vec exactVec;
    VecDuplicate(u, &exactVec);
    CHKERRQ(ierr);
    DMProjectFunction(dm, crtime, &exactFuncs[0], &exactCtxs[0], INSERT_ALL_VALUES, exactVec);
    CHKERRQ(ierr);

    // Compute the error
    VecAXPY(exactVec, -1.0,u );
    CHKERRQ(ierr);
    VecSetBlockSize(exactVec, numberOfFields);
    CHKERRQ(ierr);
    PetscInt size;
    VecGetSize(exactVec, &size);
    CHKERRQ(ierr);

    // Compute the l2 errors
    // Store the errors
    std::vector<PetscReal> ferrors(numberOfFields);
    VecStrideNormAll(exactVec, NORM_2, &ferrors[0]);
    CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g", (int)step, (double)crtime, (double)ferrors[0]);
    CHKERRQ(ierr);

    // Now print the other errors
    for (auto i = 1; i < numberOfFields; i++) {
        ierr = PetscPrintf(PETSC_COMM_WORLD, ", %2.3g", (double)ferrors[i]);
        CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "]\n");
    CHKERRQ(ierr);

    VecDestroy(&exactVec);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::Monitor, ablate::monitors::ErrorMonitor, "Computes and reports the error every time step");

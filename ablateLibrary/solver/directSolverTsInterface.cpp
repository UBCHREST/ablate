#include "directSolverTsInterface.hpp"
PetscErrorCode ablate::solver::DirectSolverTsInterface::PreStage(TS ts, PetscReal stagetime) {
    PetscFunctionBeginUser;
    ablate::solver::Solver* solver;
    TSGetApplicationContext(ts, &solver);
    try {
        solver->PreStage(ts, stagetime);
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::DirectSolverTsInterface::PreStep(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::Solver* solver;
    TSGetApplicationContext(ts, &solver);
    try {
        solver->PreStep(ts);
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::DirectSolverTsInterface::PostStep(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::Solver* solver;
    TSGetApplicationContext(ts, &solver);
    try {
        solver->PostStep(ts);
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::DirectSolverTsInterface::PostEvaluate(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::Solver* solver;
    TSGetApplicationContext(ts, &solver);
    try {
        solver->PostEvaluate(ts);
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::DirectSolverTsInterface::SetupSolverTS(std::shared_ptr<Solver> solver, TS ts) {
    PetscFunctionBeginUser;

    void* test = NULL;
    PetscErrorCode ierr = TSGetApplicationContext(ts, &test);
    CHKERRQ(ierr);

    if (test != nullptr) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "The SetupSolverTS can only be used with one solver. Please use a ablate::solver::TimeStepper");
    }

    TSSetDM(ts, solver->GetSubDomain().GetDM());
    TSSetApplicationContext(ts, solver.get());
    TSSetPreStep(ts, PreStep);
    TSSetPreStage(ts, PreStage);
    TSSetPostStep(ts, PostStep);
    TSSetPostEvaluate(ts, PostEvaluate);

    PetscFunctionReturn(0);
}

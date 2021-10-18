#include "directSolverTsInterface.hpp"
#include <utilities/petscError.hpp>

ablate::solver::DirectSolverTsInterface::DirectSolverTsInterface(TS ts, std::vector<std::shared_ptr<Solver>> solvers) : solvers(solvers) {
    void* test = NULL;
    TSGetApplicationContext(ts, &test) >> checkError;

    if (test != nullptr) {
        throw std::runtime_error("A TS can only be used with one DirectSolverTsInterface and the application contest must be DirectSolverTsInterface.");
    }

    TSSetDM(ts, solvers.front()->GetSubDomain().GetDM());
    TSSetApplicationContext(ts, this);
    TSSetPreStep(ts, PreStep);
    TSSetPreStage(ts, PreStage);
    TSSetPostStep(ts, PostStep);
    TSSetPostEvaluate(ts, PostEvaluate);
}

ablate::solver::DirectSolverTsInterface::DirectSolverTsInterface(TS ts, std::shared_ptr<Solver> solver) : DirectSolverTsInterface(ts, std::vector<std::shared_ptr<Solver>>{solver}) {}

PetscErrorCode ablate::solver::DirectSolverTsInterface::PreStage(TS ts, PetscReal stagetime) {
    PetscFunctionBeginUser;
    ablate::solver::DirectSolverTsInterface* interface;
    TSGetApplicationContext(ts, &interface);
    try {
        for (auto& solver : interface->solvers) {
            solver->PreStage(ts, stagetime);
        }
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::DirectSolverTsInterface::PreStep(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::DirectSolverTsInterface* interface;
    TSGetApplicationContext(ts, &interface);
    try {
        for (auto& solver : interface->solvers) {
            solver->PreStep(ts);
        }
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::DirectSolverTsInterface::PostStep(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::DirectSolverTsInterface* interface;
    TSGetApplicationContext(ts, &interface);
    try {
        for (auto& solver : interface->solvers) {
            solver->PostStep(ts);
        }
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::DirectSolverTsInterface::PostEvaluate(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::DirectSolverTsInterface* interface;
    TSGetApplicationContext(ts, &interface);
    try {
        for (auto& solver : interface->solvers) {
            solver->PostEvaluate(ts);
        }
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
    }
    PetscFunctionReturn(0);
}
#include "directSolverTsInterface.hpp"
#include <utilities/petscError.hpp>
#include "boundaryFunction.hpp"
#include "iFunction.hpp"



ablate::solver::DirectSolverTsInterface::DirectSolverTsInterface(TS ts, std::vector<std::shared_ptr<Solver>> solvers) : solvers(solvers) {
    void* test = NULL;
    TSGetApplicationContext(ts, &test) >> checkError;

    if (test != nullptr) {
        throw std::runtime_error("A TS can only be used with one DirectSolverTsInterface and the application contest must be DirectSolverTsInterface.");
    }

    auto dm = solvers.front()->GetSubDomain().GetDM();
    TSSetDM(ts, dm);
    TSSetApplicationContext(ts, this);
    TSSetPreStep(ts, PreStep);
    TSSetPreStage(ts, PreStage);
    TSSetPostStep(ts, PostStep);
    TSSetPostEvaluate(ts, PostEvaluate);

    if(AnyOfType<IFunction>(solvers)){
        DMTSSetIFunctionLocal(dm, ComputeIFunction, this) >> checkError;
        DMTSSetIJacobianLocal(dm, ComputeIJacobian, this) >> checkError;
    }

    if(AnyOfType<BoundaryFunction>(solvers)){
        DMTSSetBoundaryLocal(dm, ComputeBoundary, this) >> checkError;
    }
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

PetscErrorCode ablate::solver::DirectSolverTsInterface::ComputeIFunction(DM, PetscReal time, Vec locX, Vec locX_t, Vec locF, void* ctx) {
    PetscFunctionBeginUser;
    auto interface = (ablate::solver::DirectSolverTsInterface*)ctx;
    for (auto& solver : interface->solvers) {
        if(auto iFunctionSolver = std::dynamic_pointer_cast<IFunction>(solver)) {
            PetscErrorCode ierr = iFunctionSolver->ComputeIFunction(time, locX, locX_t, locF);
            CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::DirectSolverTsInterface::ComputeIJacobian(DM, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *ctx) {
    PetscFunctionBeginUser;
    auto interface = (ablate::solver::DirectSolverTsInterface*)ctx;
    for (auto& solver : interface->solvers) {
        if(auto iFunctionSolver = std::dynamic_pointer_cast<IFunction>(solver)) {
            PetscErrorCode ierr = iFunctionSolver->ComputeIJacobian(time, locX, locX_t, X_tShift, Jac, JacP);
            CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::DirectSolverTsInterface::ComputeBoundary(DM, PetscReal time, Vec locX, Vec locX_t, void *ctx) {
    PetscFunctionBeginUser;
    auto interface = (ablate::solver::DirectSolverTsInterface*)ctx;
    for (auto& solver : interface->solvers) {
        if(auto iFunctionSolver = std::dynamic_pointer_cast<BoundaryFunction>(solver)) {
            PetscErrorCode ierr = iFunctionSolver->ComputeBoundary(time, locX, locX_t);
            CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}
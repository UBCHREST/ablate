#include "timeStepper.hpp"
#include <petscdm.h>
#include <mathFunctions/mathFunction.hpp>
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::solver::TimeStepper::TimeStepper(std::string nameIn, std::shared_ptr<ablate::domain::Domain> domain, std::map<std::string, std::string> arguments,
                                         std::shared_ptr<ablate::io::Serializer> serializerIn)
    : name(nameIn), domain(domain), serializer(serializerIn) {
    // create an instance of the ts
    TSCreate(PETSC_COMM_WORLD, &ts) >> checkError;

    // force the time step to end at the exact time step
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> checkError;
    TSSetProblemType(ts, TS_NONLINEAR) >> checkError;

    // set the name and prefix as provided
    PetscObjectSetName((PetscObject)ts, name.c_str()) >> checkError;
    TSSetOptionsPrefix(ts, name.c_str()) >> checkError;

    // append any prefix values
    ablate::utilities::PetscOptionsUtils::Set(name, arguments);

    // Set this as the context
    TSSetApplicationContext(ts, this) >> checkError;

    // register the serializer with the ts
    if (serializer) {
        TSMonitorSet(ts, serializer->GetSerializeFunction(), serializer->GetContext(), NULL) >> checkError;
    }

    // Set the pre/post stage functions
    TSSetPreStage(ts, TSPreStageFunction) >> checkError;
    TSSetPreStep(ts, TSPreStepFunction) >> checkError;
    TSSetPostStep(ts, TSPostStepFunction) >> checkError;
    TSSetPostEvaluate(ts, TSPostEvaluateFunction) >> checkError;
}

ablate::solver::TimeStepper::~TimeStepper() { TSDestroy(&ts); }

void ablate::solver::TimeStepper::Solve() {
    if (solvers.empty()) {
        throw std::runtime_error("No solvers have been set.");
    }

    domain->InitializeSubDomains(solvers);
    TSSetDM(ts, domain->GetDM()) >> checkError;

    // Register any functions with the dm/ts
    if (!boundaryFunctionSolvers.empty()) {
        DMTSSetBoundaryLocal(domain->GetDM(), SolverComputeBoundaryFunctionLocal, this) >> checkError;
    }
    if (!rhsFunctionSolvers.empty()) {
        DMTSSetRHSFunctionLocal(domain->GetDM(), SolverComputeRHSFunctionLocal, this) >> checkError;
    }
    if (!iFunctionSolvers.empty()) {
        DMTSSetIFunctionLocal(domain->GetDM(), SolverComputeIFunctionLocal, this) >> checkError;
        DMTSSetIJacobianLocal(domain->GetDM(), SolverComputeIJacobianLocal, this) >> checkError;
    }

    // Register the monitors
    for (auto& solver : solvers) {
        // Get any monitors
        auto& monitorsList = monitors[solver->GetId()];
        for (auto& monitor : monitorsList) {
            monitor->Register(solver);
        }
    }
    // Register the solver with the serializer
    if (serializer) {
        for (auto& solver : solvers) {
            serializer->Register(solver);
        }
    }

    // Get the solution vector
    Vec solutionVec = domain->GetSolutionVector();

    // set the ts from options
    TSSetFromOptions(ts) >> checkError;
    TSSetSolution(ts, solutionVec) >> checkError;

    // If there was a serializer, restore the ts
    if (serializer) {
        serializer->RestoreTS(ts);
    }

    TSViewFromOptions(ts, NULL, "-ts_view") >> checkError;

    // Register the dof for the event
    PetscInt dof;
    VecGetSize(solutionVec, &dof) >> checkError;

    // create a log event
    auto logEvent = RegisterEvent(this->name.c_str());
    PetscLogEventSetDof(logEvent, 0, dof) >> checkError;
    PetscLogEventBegin(logEvent, 0, 0, 0, 0);
    TSSolve(ts, solutionVec) >> checkError;
    PetscLogEventEnd(logEvent, 0, 0, 0, 0);
}

double ablate::solver::TimeStepper::GetTime() const {
    PetscReal time;
    TSGetTime(ts, &time) >> checkError;
    return (double)time;
}

void ablate::solver::TimeStepper::Register(std::shared_ptr<ablate::solver::Solver> solver, std::vector<std::shared_ptr<monitors::Monitor>> solverMonitors) {
    // Save the solver and setup the domain
    solvers.push_back(solver);

    // Register the monitors
    for (auto& monitor : solverMonitors) {
        // store a reference to the monitor
        monitors[solver->GetId()].push_back(monitor);

        // register the monitor with the ts
        TSMonitorSet(ts, monitor->GetPetscFunction(), monitor->GetContext(), NULL) >> checkError;
    }

    // check to see if the solver implements a solver function
    if (auto interface = std::dynamic_pointer_cast<IFunction>(solver)) {
        iFunctionSolvers.push_back(interface);
    }
    if (auto interface = std::dynamic_pointer_cast<RHSFunction>(solver)) {
        rhsFunctionSolvers.push_back(interface);
    }
    if (auto interface = std::dynamic_pointer_cast<BoundaryFunction>(solver)) {
        boundaryFunctionSolvers.push_back(interface);
    }
}

PetscErrorCode ablate::solver::TimeStepper::TSPreStepFunction(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::TimeStepper* timeStepper;
    PetscErrorCode ierr = TSGetApplicationContext(ts, &timeStepper);
    CHKERRQ(ierr);

    for (auto& solver : timeStepper->solvers) {
        try {
            solver->PreStep(ts);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::TSPreStageFunction(TS ts, PetscReal stagetime) {
    PetscFunctionBeginUser;
    ablate::solver::TimeStepper* timeStepper;
    PetscErrorCode ierr = TSGetApplicationContext(ts, &timeStepper);
    CHKERRQ(ierr);

    for (const auto& solver : timeStepper->solvers) {
        try {
            solver->PreStage(ts, stagetime);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::TSPostStepFunction(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::TimeStepper* timeStepper;
    PetscErrorCode ierr = TSGetApplicationContext(ts, &timeStepper);
    CHKERRQ(ierr);

    for (const auto& solver : timeStepper->solvers) {
        try {
            solver->PostStep(ts);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::TSPostEvaluateFunction(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::TimeStepper* timeStepper;
    PetscErrorCode ierr = TSGetApplicationContext(ts, &timeStepper);
    CHKERRQ(ierr);

    for (const auto& solver : timeStepper->solvers) {
        try {
            solver->PostEvaluate(ts);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::SolverComputeBoundaryFunctionLocal(DM, PetscReal time, Vec locX, Vec locX_t, void* timeStepperCtx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    auto timeStepper = (ablate::solver::TimeStepper*)timeStepperCtx;
    for (auto& solver : timeStepper->boundaryFunctionSolvers) {
        ierr = solver->ComputeBoundary(time, locX, locX_t);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::SolverComputeIFunctionLocal(DM, PetscReal time, Vec locX, Vec locX_t, Vec locF, void* timeStepperCtx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    auto timeStepper = (ablate::solver::TimeStepper*)timeStepperCtx;
    for (auto& solver : timeStepper->iFunctionSolvers) {
        ierr = solver->ComputeIFunction(time, locX, locX_t, locF);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::SolverComputeIJacobianLocal(DM, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void* timeStepperCtx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    auto timeStepper = (ablate::solver::TimeStepper*)timeStepperCtx;
    for (auto& solver : timeStepper->iFunctionSolvers) {
        ierr = solver->ComputeIJacobian(time, locX, locX_t, X_tShift, Jac, JacP);
        CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::TimeStepper::SolverComputeRHSFunctionLocal(DM dm, PetscReal time, Vec locX, Vec F, void* timeStepperCtx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    auto timeStepper = (ablate::solver::TimeStepper*)timeStepperCtx;

    Vec locF;
    DMGetLocalVector(dm, &locF);
    VecZeroEntries(locF);

    for (auto& solver : timeStepper->rhsFunctionSolvers) {
        ierr = solver->ComputeRHSFunction(time, locX, locF);
        CHKERRQ(ierr);
    }

    DMLocalToGlobalBegin(dm, locF, ADD_VALUES, F);
    DMLocalToGlobalEnd(dm, locF, ADD_VALUES, F);

    DMRestoreLocalVector(dm, &locF);

    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTERDEFAULT(ablate::solver::TimeStepper, ablate::solver::TimeStepper, "the basic stepper", ARG(std::string, "name", "the time stepper name"),
                ARG(ablate::domain::Domain, "domain", "the mesh used for the simulation"), ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"),
                OPT(ablate::io::Serializer, "io", "the serializer used with this timestepper"));

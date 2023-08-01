#include "timeStepper.hpp"
#include <petscdm.h>
#include <utility>
#include "adaptPhysics.hpp"
#include "adaptPhysicsConstrained.hpp"
#include "utilities/petscUtilities.hpp"

ablate::solver::TimeStepper::TimeStepper(std::shared_ptr<ablate::domain::Domain> domain, const std::shared_ptr<ablate::parameters::Parameters>& arguments, std::shared_ptr<io::Serializer> serializer,
                                         std::shared_ptr<ablate::domain::Initializer> initializations, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions,
                                         std::vector<std::shared_ptr<mathFunctions::FieldFunction>> absoluteTolerances, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> relativeTolerances,
                                         bool verboseSourceCheck)
    : ablate::solver::TimeStepper::TimeStepper("", std::move(domain), arguments, std::move(serializer), std::move(initializations), std::move(exactSolutions), std::move(absoluteTolerances),
                                               std::move(relativeTolerances), verboseSourceCheck) {}

ablate::solver::TimeStepper::TimeStepper(const std::string& nameIn, std::shared_ptr<ablate::domain::Domain> domain, const std::shared_ptr<ablate::parameters::Parameters>& arguments,
                                         std::shared_ptr<ablate::io::Serializer> serializerIn, std::shared_ptr<ablate::domain::Initializer> initializations,
                                         std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> absoluteTolerances,
                                         std::vector<std::shared_ptr<mathFunctions::FieldFunction>> relativeTolerances, bool verboseSourceCheck)
    : utilities::StaticInitializer([] {
          AdaptPhysics::Register();
          AdaptPhysicsConstrained::Register();
      }),
      name(nameIn.empty() ? "timeStepper" : nameIn),
      domain(std::move(domain)),
      serializer(std::move(serializerIn)),
      verboseSourceCheck(verboseSourceCheck),
      initializations(std::move(initializations)),
      exactSolutions(std::move(exactSolutions)),
      absoluteTolerances(std::move(absoluteTolerances)),
      relativeTolerances(std::move(relativeTolerances))

{
    // create an instance of the ts
    TSCreate(PETSC_COMM_WORLD, &ts) >> utilities::PetscUtilities::checkError;

    // force the time step to end at the exact time step
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> utilities::PetscUtilities::checkError;
    TSSetProblemType(ts, TS_NONLINEAR) >> utilities::PetscUtilities::checkError;

    // set the name and prefix as provided
    PetscObjectSetName((PetscObject)ts, name.c_str()) >> utilities::PetscUtilities::checkError;

    // append any prefix values
    if (arguments) {
        auto argumentMap = arguments->ToMap<std::string>();
        ablate::utilities::PetscUtilities::Set(name, argumentMap);
        // only set the option prefix if an argument was provided, else use global options
        TSSetOptionsPrefix(ts, name.c_str()) >> utilities::PetscUtilities::checkError;
    }
    // Set this as the context
    TSSetApplicationContext(ts, this) >> utilities::PetscUtilities::checkError;

    // register the serializer with the ts
    if (serializer) {
        TSMonitorSet(ts, serializer->GetSerializeFunction(), serializer->GetContext(), nullptr) >> utilities::PetscUtilities::checkError;
    }

    // Set the pre/post stage functions
    TSSetPreStage(ts, TSPreStageFunction) >> utilities::PetscUtilities::checkError;
    TSSetPreStep(ts, TSPreStepFunction) >> utilities::PetscUtilities::checkError;
    TSSetPostStep(ts, TSPostStepFunction) >> utilities::PetscUtilities::checkError;
    TSSetPostEvaluate(ts, TSPostEvaluateFunction) >> utilities::PetscUtilities::checkError;
}

ablate::solver::TimeStepper::~TimeStepper() { TSDestroy(&ts) >> utilities::PetscUtilities::checkError; }

void ablate::solver::TimeStepper::Initialize() {
    StartEvent((this->name + "::Initialize").c_str());
    if (!initialized) {
        domain->InitializeSubDomains(solvers, initializations, exactSolutions);
        TSSetDM(ts, domain->GetDM()) >> utilities::PetscUtilities::checkError;
        initialized = true;

        // Register any functions with the dm/ts
        if (!boundaryFunctionSolvers.empty()) {
            DMTSSetBoundaryLocal(domain->GetDM(), SolverComputeBoundaryFunctionLocal, this) >> utilities::PetscUtilities::checkError;
        }
        if (!rhsFunctionSolvers.empty()) {
            DMTSSetRHSFunction(domain->GetDM(), SolverComputeRHSFunction, this) >> utilities::PetscUtilities::checkError;
        }
        if (!iFunctionSolvers.empty()) {
            DMTSSetIFunctionLocal(domain->GetDM(), SolverComputeIFunctionLocal, this) >> utilities::PetscUtilities::checkError;
            DMTSSetIJacobianLocal(domain->GetDM(), SolverComputeIJacobianLocal, this) >> utilities::PetscUtilities::checkError;
        }

        // Register the monitors
        for (auto& solver : solvers) {
            // Get any monitors
            auto& monitorsList = monitors[solver->GetSolverId()];
            for (auto& monitor : monitorsList) {
                monitor->Register(solver);
            }
        }

        if (serializer) {
            // Register any subdomain with the serializer
            for (auto& subDomain : domain->GetSerializableSubDomains()) {
                if (auto subDomainPtr = subDomain.lock()) {
                    if (subDomainPtr->Serialize()) {
                        serializer->Register(subDomain);
                    }
                }
            }

            // Register the solver with the serializer
            for (auto& solver : solvers) {
                auto serializable = std::dynamic_pointer_cast<io::Serializable>(solver);
                if (serializable && serializable->Serialize()) {
                    serializer->Register(serializable);
                }
            }

            // register any monitors with the seralizer
            for (const auto& monitorPerSolver : monitors) {
                for (const auto& monitor : monitorPerSolver.second) {
                    auto serializable = std::dynamic_pointer_cast<io::Serializable>(monitor);
                    if (serializable && serializable->Serialize()) {
                        serializer->Register(serializable);
                    }
                }
            }
        }
        // Get the solution vector
        Vec solutionVec = domain->GetSolutionVector();

        // set the ts from options
        TSSetSolution(ts, solutionVec) >> utilities::PetscUtilities::checkError;
        TSSetFromOptions(ts) >> utilities::PetscUtilities::checkError;
    }
    EndEvent();
}
void ablate::solver::TimeStepper::Solve() {
    // Call initialize, this will only initialize if it has not been called
    Initialize();

    TSAdapt adapt = nullptr;
    TSGetAdapt(ts, &adapt) >> ablate::utilities::PetscUtilities::checkError;
    if (adapt) {
        TSAdaptType adaptType;
        TSAdaptGetType(adapt, &adaptType) >> ablate::utilities::PetscUtilities::checkError;
        if (auto adaptInitializer = adaptInitializers[std::string(adaptType)]) {
            adaptInitializer(ts, adapt);
        }
    }

    // Get the solution vector
    Vec solutionVec = domain->GetSolutionVector();

    // If there was a serializer, restore the ts
    if (serializer) {
        serializer->RestoreTS(ts);
    }

    // If there are no solvers
    if (solvers.empty()) {
        // write at least one output
        if (serializer) {
            PetscInt step;
            TSGetStepNumber(ts, &step) >> utilities::PetscUtilities::checkError;
            PetscReal time;
            TSGetTime(ts, &time) >> utilities::PetscUtilities::checkError;

            serializer->Serialize(ts, step, time, domain->GetSolutionVector()) >> utilities::PetscUtilities::checkError;
            serializer->Serialize(ts, step + 1, time, domain->GetSolutionVector()) >> utilities::PetscUtilities::checkError;
        }
        // exit before ts solver
        return;
    }

    // set time stepper individual tolerances if specified
    if (!absoluteTolerances.empty() || !relativeTolerances.empty()) {
        Vec vatol = nullptr;
        Vec vrtol = nullptr;

        DMCreateGlobalVector(domain->GetDM(), &vatol) >> utilities::PetscUtilities::checkError;
        DMCreateGlobalVector(domain->GetDM(), &vrtol) >> utilities::PetscUtilities::checkError;

        // Get the default values
        PetscReal aTolDefault, rTolDefault;
        TSGetTolerances(ts, &aTolDefault, nullptr, &rTolDefault, nullptr) >> utilities::PetscUtilities::checkError;

        // Set the default values
        VecSet(vatol, aTolDefault) >> utilities::PetscUtilities::checkError;
        VecSet(vrtol, rTolDefault) >> utilities::PetscUtilities::checkError;

        // project the tolerances
        domain->ProjectFieldFunctions(absoluteTolerances, vatol);
        domain->ProjectFieldFunctions(relativeTolerances, vrtol);

        // Set the values
        TSSetTolerances(ts, PETSC_DECIDE, vatol, PETSC_DECIDE, vrtol) >> utilities::PetscUtilities::checkError;
        VecDestroy(&vatol);
        VecDestroy(&vrtol);
    }

    TSViewFromOptions(ts, nullptr, "-ts_view") >> utilities::PetscUtilities::checkError;

    // Register the dof for the event
    PetscInt dof;
    VecGetSize(solutionVec, &dof) >> utilities::PetscUtilities::checkError;

    // create a log event
    auto logEvent = RegisterEvent((this->name + "::Solve").c_str());
    PetscLogEventSetDof(logEvent, 0, dof) >> utilities::PetscUtilities::checkError;
    PetscLogEventBegin(logEvent, 0, 0, 0, 0);
    TSSolve(ts, solutionVec) >> utilities::PetscUtilities::checkError;
    PetscLogEventEnd(logEvent, 0, 0, 0, 0);
}

double ablate::solver::TimeStepper::GetTime() const {
    PetscReal time;
    TSGetTime(ts, &time) >> utilities::PetscUtilities::checkError;
    return (double)time;
}

void ablate::solver::TimeStepper::Register(const std::shared_ptr<ablate::solver::Solver>& solver, const std::vector<std::shared_ptr<monitors::Monitor>>& solverMonitors) {
    // Save the solver and setup the domain
    solvers.push_back(solver);

    // Register the monitors
    for (auto& monitor : solverMonitors) {
        // store a reference to the monitor
        monitors[solver->GetSolverId()].push_back(monitor);

        // register the monitor with the ts
        if (auto monitorFunction = monitor->GetPetscFunction()) {
            TSMonitorSet(ts, monitorFunction, monitor->GetContext(), nullptr) >> utilities::PetscUtilities::checkError;
        }
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
    if (auto interface = std::dynamic_pointer_cast<PhysicsTimeStepFunction>(solver)) {
        physicsTimeStepFunctionSolvers.push_back(interface);
    }
}

PetscErrorCode ablate::solver::TimeStepper::TSPreStepFunction(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::TimeStepper* timeStepper;
    PetscCall(TSGetApplicationContext(ts, &timeStepper));

    for (auto& solver : timeStepper->solvers) {
        try {
            solver->PreStep(ts);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::TSPreStageFunction(TS ts, PetscReal stagetime) {
    PetscFunctionBeginUser;

    // Store if we are in the first stage
    PetscReal time;
    PetscCall(TSGetTime(ts, &time));

    // only continue if the stage time is the real time (i.e. the first stage)
    ablate::solver::TimeStepper* timeStepper;
    PetscCall(TSGetApplicationContext(ts, &timeStepper));
    // Set to try if time == stagetime
    timeStepper->runInitialStep = timeStepper->runInitialStep || (time == stagetime);

    for (const auto& solver : timeStepper->solvers) {
        try {
            solver->PreStage(ts, stagetime);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::TSPostStepFunction(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::TimeStepper* timeStepper;
    PetscCall(TSGetApplicationContext(ts, &timeStepper));

    for (const auto& solver : timeStepper->solvers) {
        try {
            solver->PostStep(ts);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::TSPostEvaluateFunction(TS ts) {
    PetscFunctionBeginUser;
    ablate::solver::TimeStepper* timeStepper;
    PetscCall(TSGetApplicationContext(ts, &timeStepper));

    for (const auto& solver : timeStepper->solvers) {
        try {
            solver->PostEvaluate(ts);
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::SolverComputeBoundaryFunctionLocal(DM, PetscReal time, Vec locX, Vec locX_t, void* timeStepperCtx) {
    PetscFunctionBeginUser;
    auto timeStepper = (ablate::solver::TimeStepper*)timeStepperCtx;
    for (auto& solver : timeStepper->boundaryFunctionSolvers) {
        PetscCall(solver->ComputeBoundary(time, locX, locX_t));
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::SolverComputeIFunctionLocal(DM, PetscReal time, Vec locX, Vec locX_t, Vec locF, void* timeStepperCtx) {
    PetscFunctionBeginUser;

    auto timeStepper = (ablate::solver::TimeStepper*)timeStepperCtx;
    for (auto& solver : timeStepper->iFunctionSolvers) {
        PetscCall(solver->ComputeIFunction(time, locX, locX_t, locF));
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::SolverComputeIJacobianLocal(DM, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void* timeStepperCtx) {
    PetscFunctionBeginUser;

    auto timeStepper = (ablate::solver::TimeStepper*)timeStepperCtx;
    for (auto& solver : timeStepper->iFunctionSolvers) {
        PetscCall(solver->ComputeIJacobian(time, locX, locX_t, X_tShift, Jac, JacP));
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::solver::TimeStepper::SolverComputeRHSFunction(TS ts, PetscReal time, Vec X, Vec F, void* timeStepperCtx) {
    PetscFunctionBeginUser;
    auto timeStepper = (ablate::solver::TimeStepper*)timeStepperCtx;

    timeStepper->StartEvent("SolverComputeRHSFunction::DMGlobalToLocal");
    DM dm = timeStepper->domain->GetDM();
    Vec locX, locF;
    DMGetLocalVector(dm, &locX);
    DMGetLocalVector(dm, &locF);
    VecZeroEntries(locX);

    // Fill the ghost nodes (and all others).  Note the boundary/local field is swapped from the petsc version
    DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX);
    DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX);
    timeStepper->EndEvent();

    // Update the boundary conditions
    timeStepper->StartEvent("SolverComputeRHSFunction::SolverComputeBoundaryFunctionLocal");
    PetscCall(SolverComputeBoundaryFunctionLocal(dm, time, locX, nullptr, timeStepperCtx));
    timeStepper->EndEvent();

    // Call each of the provided pre RHS functions
    timeStepper->StartEvent("SolverComputeRHSFunction::PreRHSFunction");
    for (auto& solver : timeStepper->rhsFunctionSolvers) {
        PetscCall(solver->PreRHSFunction(ts, time, timeStepper->runInitialStep, locX));
    }
    timeStepper->EndEvent();

    // Reset the timeStepper->runInitialStep
    timeStepper->runInitialStep = false;

    // Zero out the temp locF array
    VecZeroEntries(locF);
    CHKMEMQ;

    // Call each of the provided RHS functions
    timeStepper->StartEvent("SolverComputeRHSFunction::ComputeRHSFunction");
    for (auto& solver : timeStepper->rhsFunctionSolvers) {
        PetscCall(solver->ComputeRHSFunction(time, locX, locF));
    }
    CHKMEMQ;
    timeStepper->EndEvent();

    timeStepper->StartEvent("SolverComputeRHSFunction::DMLocalToGlobalEnd");
    VecZeroEntries(F);
    DMLocalToGlobalBegin(dm, locF, ADD_VALUES, F);
    DMLocalToGlobalEnd(dm, locF, ADD_VALUES, F);
    DMRestoreLocalVector(dm, &locX);
    DMRestoreLocalVector(dm, &locF);
    timeStepper->EndEvent();

    if (timeStepper->verboseSourceCheck) {
        timeStepper->StartEvent("SolverComputeRHSFunction::CheckFieldValues");
        timeStepper->domain->CheckFieldValues(F);
        timeStepper->EndEvent();
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::solver::TimeStepper::ComputePhysicsTimeStep(PetscReal* dt) {
    PetscFunctionBeginUser;
    PetscReal localDtMin = PETSC_INFINITY;
    try {
        for (auto& timeStepFunction : physicsTimeStepFunctionSolvers) {
            localDtMin = PetscMin(timeStepFunction->ComputePhysicsTimeStep(ts), localDtMin);
        }
    } catch (std::exception& exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
    }

    // Take the global min
    PetscCallMPI(MPI_Allreduce(&localDtMin, dt, 1, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)ts)));

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::solver::TimeStepper, ablate::solver::TimeStepper, "the basic stepper", OPT(std::string, "name", "the optional time stepper name"),
                 ARG(ablate::domain::Domain, "domain", "the mesh used for the simulation"), OPT(ablate::parameters::Parameters, "arguments", "arguments to be passed to petsc"),
                 OPT(ablate::io::Serializer, "io", "the serializer used with this timestepper"), OPT(ablate::domain::Initializer, "initialization", "initialization field functions"),
                 OPT(std::vector<ablate::mathFunctions::FieldFunction>, "exactSolution", "optional exact solutions that can be used for error calculations"),
                 OPT(std::vector<ablate::mathFunctions::FieldFunction>, "absoluteTolerances", "optional absolute tolerances for a field"),
                 OPT(std::vector<ablate::mathFunctions::FieldFunction>, "relativeTolerances", "optional relative tolerances for a field"),
                 OPT(bool, "verboseSourceCheck", "does a slow nan/inf for solvers that use rhs evaluation. This is slow and should only be used for debug."));

#include "timeStepper.hpp"
#include <petscdm.h>
#include <mathFunctions/mathFunction.hpp>
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::solve::TimeStepper::TimeStepper(std::string nameIn, std::map<std::string, std::string> arguments) : name(nameIn), tsLogEvent(-1) {
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

    // Setup the tsSolve event for this solve
    std::string eventName = name + "::TSSolve";
    PetscLogEventGetId(eventName.c_str(), &tsLogEvent) >> checkError;
    if (tsLogEvent < 0) {
        PetscLogEventRegister(eventName.c_str(), GetPetscClassId(), &tsLogEvent) >> checkError;
    }
}

ablate::solve::TimeStepper::~TimeStepper() { TSDestroy(&ts); }

void ablate::solve::TimeStepper::Solve(std::shared_ptr<Solvable> solvable, std::shared_ptr<ablate::environment::RestartManager> restartManagerIn) {
    // Get the solution vector
    Vec solutionVec = solvable->GetSolutionVector();

    // set the ts from options
    TSSetFromOptions(ts) >> checkError;
    TSSetSolution(ts, solutionVec) >> checkError;

    // If there are restart parameters, update the ts
    if (restartManagerIn) {
        restartManager = restartManagerIn;

        // register the restart with the ts
        TSMonitorSet(ts, restartManager->GetTSFunction(), restartManager->GetContext(), NULL) >> checkError;

        // pass a weak pointer ot the restart manager
        restartManager->Register(shared_from_this());
    }

    TSViewFromOptions(ts, NULL, "-ts_view") >> checkError;

    // Register the dof for the event
    PetscInt dof;
    VecGetSize(solutionVec, &dof) >> checkError;
    PetscLogEventSetDof(tsLogEvent, 0, dof) >> checkError;
    PetscLogEventBegin(tsLogEvent, 0, 0, 0, 0);
    TSSolve(ts, solutionVec) >> checkError;
    PetscLogEventEnd(tsLogEvent, 0, 0, 0, 0);
}

void ablate::solve::TimeStepper::AddMonitor(std::shared_ptr<monitors::Monitor> monitor) {
    // store a reference to the monitor
    monitors.push_back(monitor);

    // register the monitor with the ts
    TSMonitorSet(ts, monitor->GetPetscFunction(), monitor->GetContext(), NULL) >> checkError;
}

double ablate::solve::TimeStepper::GetTime() const {
    PetscReal time;
    TSGetTime(ts, &time) >> checkError;
    return (double)time;
}

PetscClassId ablate::solve::TimeStepper::GetPetscClassId() {
    // Register this class with petsc if flow has not been register
    if (!petscClassId) {
        PetscClassIdRegister("ablate::solve::TimeStepper", &petscClassId) >> checkError;
    }
    return petscClassId;
}

void ablate::solve::TimeStepper::Save(environment::SaveState& saveState) const {
    PetscReal time;
    TSGetTime(ts, &time) >> checkError;
    saveState.Save("time", time);

    PetscReal dt;
    TSGetTimeStep(ts, &dt) >> checkError;
    saveState.Save("dt", dt);

    PetscInt steps;
    TSGetStepNumber(ts, &steps) >> checkError;
    saveState.Save("steps", steps);

    Vec solutionVec;
    TSGetSolution(ts, &solutionVec )>> checkError;;
    saveState.Save("solutionVec", solutionVec);
}

void ablate::solve::TimeStepper::Restore(const environment::RestoreState& restoreState) {

    TSSetStepNumber(ts, restoreState.GetExpect<PetscInt>("steps"));
    TSSetTime(ts, restoreState.GetExpect<PetscReal>("time"));
    TSSetTimeStep(ts, restoreState.GetExpect<PetscReal>("dt"));

    // Load the saved vector
    Vec solutionVec;
    TSGetSolution(ts, &solutionVec )>> checkError;;
    restoreState.Get("solutionVec", solutionVec);
}

REGISTERDEFAULT(ablate::solve::TimeStepper, ablate::solve::TimeStepper, "the basic stepper", ARG(std::string, "name", "the time stepper name"),
                ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"));

#include "timeStepper.hpp"
#include <petscdm.h>
#include <mathFunctions/mathFunction.hpp>
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::solve::TimeStepper::TimeStepper(std::string nameIn, std::map<std::string, std::string> arguments, std::shared_ptr<ablate::io::Serializer> serializerIn)
    : name(nameIn), tsLogEvent(-1), serializer(serializerIn) {
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

    // register the serializer with the ts
    if(serializer) {
        TSMonitorSet(ts, serializer->GetSerializeFunction(), serializer->GetContext(), NULL) >> checkError;
    }
}

ablate::solve::TimeStepper::~TimeStepper() { TSDestroy(&ts); }

void ablate::solve::TimeStepper::Solve(std::shared_ptr<Solvable> solvable) {
    // Get the solution vector
    Vec solutionVec = solvable->GetSolutionVector();

    // set the ts from options
    TSSetFromOptions(ts) >> checkError;
    TSSetSolution(ts, solutionVec) >> checkError;

    // If there was a serializer, restore the ts
    if(serializer) {
        serializer->RestoreTS(ts);
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

void ablate::solve::TimeStepper::Register(std::weak_ptr<io::Serializable> serializable) {
    if(serializer){
        serializer->Register(serializable);
    }
}

REGISTERDEFAULT(ablate::solve::TimeStepper, ablate::solve::TimeStepper, "the basic stepper", ARG(std::string, "name", "the time stepper name"),
                ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"),
                OPT(ablate::io::Serializer, "io", "the serializer used with this timestepper"));

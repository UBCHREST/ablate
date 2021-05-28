#include "timeStepper.hpp"
#include <petscdm.h>
#include <mathFunctions/mathFunction.hpp>
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    DM dm;
    PetscDS ds;
    Vec v;

    PetscFunctionBeginUser;
    TSGetDM(ts, &dm) >> ablate::checkError;
    DMGetDS(dm, &ds) >> ablate::checkError;

    // Get the number of fields
    PetscInt numberOfFields;
    PetscDSGetNumFields(ds, &numberOfFields) >> ablate::checkError;

    // Get the exact funcs and contx
    std::vector<ablate::mathFunctions::PetscFunction> functions(numberOfFields);
    std::vector<void *> ctxs(numberOfFields);
    for (auto f = 0; f < numberOfFields; ++f) {
        PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]) >> ablate::checkError;
    }

    // Store the errors
    std::vector<PetscReal> ferrors(numberOfFields);
    DMComputeL2FieldDiff(dm, crtime, &exactFuncs[0], &ctxs[0], u, &ferrors[0]) >> ablate::checkError;

    PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g", (int)step, (double)crtime, (double)ferrors[0]) >> ablate::checkError;

    // Now print the other errors
    for (auto i = 1; i < numberOfFields; i++) {
        PetscPrintf(PETSC_COMM_WORLD, ", %2.3g", (double)ferrors[i]) >> ablate::checkError;
    }
    PetscPrintf(PETSC_COMM_WORLD, "]\n") >> ablate::checkError;

    PetscFunctionReturn(0);
}

ablate::solve::TimeStepper::TimeStepper(std::string name, std::map<std::string, std::string> arguments) : comm(PETSC_COMM_WORLD), name(name) {
    // create an instance of the ts
    TSCreate(PETSC_COMM_WORLD, &ts) >> checkError;

    // force the time step to end at the exact time step
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> checkError;

    // set the name and prefix as provided
    PetscObjectSetName((PetscObject)ts, name.c_str()) >> checkError;
    TSSetOptionsPrefix(ts, name.c_str()) >> checkError;

    // append any prefix values
    ablate::utilities::PetscOptionsUtils::Set(name, arguments);

    // Set this as the context
    TSSetApplicationContext(ts, this) >> checkError;
}

ablate::solve::TimeStepper::~TimeStepper() { TSDestroy(&ts); }

void ablate::solve::TimeStepper::Solve(std::shared_ptr<Solvable> solvable) {
    // Get the solution vector
    Vec solutionVec = solvable->GetSolutionVector();

    // set the ts from options
    TSSetFromOptions(ts) >> checkError;

    // finish setting up the ts
    PetscReal time;
    TSGetTime(ts, &time) >> checkError;

    // reset the dm
    DM dm;
    TSGetDM(ts, &dm) >> checkError;

    TSMonitorSet(ts, MonitorError, NULL, NULL) >> checkError;

    TSViewFromOptions(ts, NULL, "-ts_view") >> checkError;

    TSSolve(ts, solutionVec) >> checkError;
}
void ablate::solve::TimeStepper::AddMonitor(std::shared_ptr<monitors::Monitor> monitor) {
    // store a reference to the monitor
    monitors.push_back(monitor);

    // register the monitor with the ts
    TSMonitorSet(ts, monitor->GetPetscFunction(), monitor->GetContext(), NULL) >> checkError;
}

REGISTERDEFAULT(ablate::solve::TimeStepper, ablate::solve::TimeStepper, "the basic stepper", ARG(std::string, "name", "the time stepper name"),
                ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"));

#include "timeStepper.hpp"
#include <petscdm.h>
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) {
    PetscErrorCode (*exactFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    void *ctxs[3];
    DM dm;
    PetscDS ds;
    Vec v;
    PetscReal ferrors[3];
    PetscInt f;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    ierr = VecViewFromOptions(u, NULL, "-vec_view_monitor");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    for (f = 0; f < 3; ++f) {
        ierr = PetscDSGetExactSolution(ds, f, &exactFuncs[f], &ctxs[f]);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
    ierr = DMComputeL2FieldDiff(dm, crtime, exactFuncs, ctxs, u, ferrors);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g, %2.3g]\n", (int)step, (double)crtime, (double)ferrors[0], (double)ferrors[1], (double)ferrors[2]);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

    ierr = DMGetGlobalVector(dm, &v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    // ierr = VecSet(v, 0.0);CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = PetscObjectSetName((PetscObject)v, "Exact Solution");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecViewFromOptions(v, NULL, "-exact_vec_view");
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = DMRestoreGlobalVector(dm, &v);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);

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
    ablate::utilities::PetscOptions::Set(name, arguments);

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

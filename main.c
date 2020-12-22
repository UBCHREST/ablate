static char help[] = "Time-dependent Low Mach Flow in 2d channels with finite elements.\n\
We solve the Low Mach flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include "lowMachFlow.h"
#include "mesh.h"

int main(int argc,char **args)
{
    DM                  dm;   /* problem definition */
    TS                  ts;   /* timestepper */
    Vec                 u;    /* solution */
    LowMachFlowContext  context; /* user-defined work context */
    PetscReal           t;
    PetscErrorCode      ierr;

    // initialize petsc and mpi
    ierr = PetscInitialize(&argc, &args, NULL,help);if (ierr) return ierr;

    // setup and initialize the constant field variables
    ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &context.parameters);CHKERRQ(ierr);
    ierr = SetupParameters(&context);CHKERRQ(ierr);

    // setup the ts
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
    ierr = CreateMesh(PETSC_COMM_WORLD, &dm, PETSC_TRUE, 2);CHKERRQ(ierr);
    ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
    ierr = DMSetApplicationContext(dm, &context);CHKERRQ(ierr);

    // setup problem
    ierr = SetupDiscretization(dm, &context);CHKERRQ(ierr);
    ierr = SetupProblem(dm, &context);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);

    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetPreStep(ts, RemoveDiscretePressureNullspace);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
//    ierr = TSSetComputeInitialCondition(ts, SetInitialConditions);CHKERRQ(ierr); /* Must come after SetFromOptions() */
//    ierr = SetInitialConditions(ts, u);CHKERRQ(ierr);

    ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
    ierr = DMSetOutputSequenceNumber(dm, 0, t);CHKERRQ(ierr);
    ierr = DMTSCheckFromOptions(ts, u);CHKERRQ(ierr);
//    ierr = TSMonitorSet(ts, MonitorError, &user, NULL);CHKERRQ(ierr);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);
    ierr = TSSolve(ts, u);CHKERRQ(ierr);

    // Cleanup
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
    ierr = PetscBagDestroy(&context.parameters);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}
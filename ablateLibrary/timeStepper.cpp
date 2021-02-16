//
// Created by Matt McGurn on 2/15/21.
//

#include "timeStepper.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::TimeStepper::TimeStepper(MPI_Comm comm, std::string name, std::map<std::string, std::string> arguments):
    comm(comm),name(name)
{
    // create an instance of the ts
    TSCreate(PETSC_COMM_WORLD, &ts) >> checkError;

    // force the time step to end at the exact time step
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP) >> checkError;

    // set the name and prefix as provided
    PetscObjectSetName((PetscObject)ts, name.c_str()) >> checkError;
    TSSetOptionsPrefix(ts, name.c_str()) >> checkError;

    // append any prefix values
    ablate::utilities::PetscOptions::Set(name, arguments);

    // set the ts from options
    TSSetFromOptions(ts) >> checkError;

    // Set this as the context
    TSSetApplicationContext(ts, this) >> checkError;
}

ablate::TimeStepper::~TimeStepper() {
    TSDestroy(&ts);
}


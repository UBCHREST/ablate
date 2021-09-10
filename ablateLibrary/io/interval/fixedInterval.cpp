
#include "fixedInterval.hpp"
ablate::io::interval::FixedInterval::FixedInterval(int interval) : interval(interval) {}

bool ablate::io::interval::FixedInterval::Check(MPI_Comm comm, PetscInt steps, PetscReal time) { return steps == 0 || interval == 0 || (steps % interval == 0); }

#include "parser/registrar.hpp"
REGISTERDEFAULT_PASS_THROUGH(ablate::io::interval::Interval, ablate::io::interval::FixedInterval, "Default interval that outputs every n steps", int);

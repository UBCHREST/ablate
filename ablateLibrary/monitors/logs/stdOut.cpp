#include "stdOut.hpp"
#include <stdarg.h>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>

void ablate::monitors::logs::StdOut::Initialize(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank) >> checkMpiError;
    output = rank == 0;
}

void ablate::monitors::logs::StdOut::Printf(const char* format, ...) {
    if (output) {
        va_list args;
        va_start(args, format);
        PetscVFPrintf(PETSC_STDOUT, format, args) >> checkError;
        va_end(args);
    }
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::logs::Log, ablate::monitors::logs::StdOut, "Writes to the standard out");

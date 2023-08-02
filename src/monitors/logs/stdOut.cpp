#include "stdOut.hpp"
#include "nullLog.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

void ablate::monitors::logs::StdOut::Initialize(MPI_Comm comm) {
    Log::Initialize(comm);
    int rank;
    MPI_Comm_rank(comm, &rank) >> utilities::MpiUtilities::checkError;
    output = rank == 0;
}

void ablate::monitors::logs::StdOut::Printf(const char* format, ...) {
    if (output) {
        va_list args;
        va_start(args, format);
        PetscVFPrintf(PETSC_STDOUT, format, args) >> utilities::PetscUtilities::checkError;
        va_end(args);
    }
}
std::ostream& ablate::monitors::logs::StdOut::GetStream() {
    if (output) {
        return std::cout;
    } else {
        return NullLog::nullStream;
    }
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::monitors::logs::Log, ablate::monitors::logs::StdOut, "Writes to the standard out");

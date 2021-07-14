#ifndef ABLATELIBRARY_LOG_HPP
#define ABLATELIBRARY_LOG_HPP

#include <petsc.h>

namespace ablate::monitors::logs {
    class Log {
       public:
        virtual ~Log() = default;

        // each log must support the print command
        virtual void Printf(const char*, ...) = 0;
        virtual void Print(const char* value) {
            Printf(value);
        }
        virtual void Initialize(MPI_Comm comm = MPI_COMM_SELF) = 0;
    };
}

#endif  // ABLATELIBRARY_LOG_HPP

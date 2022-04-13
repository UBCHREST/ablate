#ifndef ABLATELIBRARY_MPIERROR_HPP
#define ABLATELIBRARY_MPIERROR_HPP
#include <petscsys.h>
#include <iostream>

namespace ablate {
namespace utilities {
class MpiErrorChecker {
   public:
    struct MpiError : public std::runtime_error {
       private:
        static std::string GetMessage(int ierr) {
            char estring[MPI_MAX_ERROR_STRING];
            int len;
            MPI_Error_string(ierr, estring, &len);
            return "MPI Error: " + std::string(estring);
        }

       public:
        MpiError(int ierr) : std::runtime_error(GetMessage(ierr)) {}
    };

    friend void operator>>(int ierr, const MpiErrorChecker &errorChecker) {
        if (MPI_SUCCESS != ierr) {
            throw MpiError(ierr);
        }
    }
};
}  // namespace utilities

inline utilities::MpiErrorChecker checkMpiError;
}  // namespace ablate

#endif  // ABLATELIBRARY_MPIERROR_HPP

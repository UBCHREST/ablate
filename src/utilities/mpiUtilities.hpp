#ifndef ABLATELIBRARY_MPIUTILITIES_HPP
#define ABLATELIBRARY_MPIUTILITIES_HPP
#include <petsc.h>
#include <functional>

namespace ablate::utilities {

class MpiUtilities {
    /**
     * helper class to check mpi errors
     */
    class ErrorChecker {
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
            explicit MpiError(int ierr) : std::runtime_error(GetMessage(ierr)) {}
        };

        inline friend void operator>>(int ierr, const ErrorChecker &errorChecker) {
            if (MPI_SUCCESS != ierr) {
                throw MpiError(ierr);
            }
        }
    };

   public:
    /**
     * call to apply in function in order one by one (useful for setup)
     * @param comm
     */
    static void RoundRobin(MPI_Comm comm, std::function<void(int rank)>);

    /**
     * call this function on root and wait to complete
     * @param comm
     */
    static void Once(MPI_Comm comm, std::function<void()>, int root = 0);

    /**
     * static inline error checker for mpi based errors
     */
    static inline utilities::MpiUtilities::ErrorChecker checkError;

   private:
    MpiUtilities() = delete;
};
}  // namespace ablate::utilities
#endif  // ABLATELIBRARY_PETSCUTILITIES_HPP

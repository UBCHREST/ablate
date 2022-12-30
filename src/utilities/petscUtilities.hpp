#ifndef ABLATELIBRARY_PETSCUTILITIES_HPP
#define ABLATELIBRARY_PETSCUTILITIES_HPP
#include <petsc.h>

namespace ablate::utilities {

class PetscUtilities {
   public:
    /**
     * helper class to check mpi errors
     */
    class ErrorChecker {
       public:
        struct PetscError : public std::runtime_error {
           private:
            static std::string GetMessage(PetscErrorCode ierr) {
                const char* text;
                char* specific;

                PetscErrorMessage(ierr, &text, &specific);

                return std::string(text) + ": " + std::string(specific);
            }

           public:
            explicit PetscError(PetscErrorCode ierr) : std::runtime_error(GetMessage(ierr)) {}
        };

        inline friend void operator>>(PetscErrorCode ierr, const ErrorChecker&) {
            if (ierr != 0) {
                throw PetscError(ierr);
            }
        }
    };

   public:
    /**
     * static call to setup petsc petsc and register cleanup call
     */
    static void Initialize(const char[] = nullptr);

    /**
     * static inline error checker for petsc based errors
     */
    static inline utilities::PetscUtilities::ErrorChecker checkError;

    PetscUtilities() = delete;
};

/**
 * public function to convert from a stream to PetscDataType
 * @param is
 * @param v
 * @return
 */
std::istream& operator>>(std::istream& is, PetscDataType& v);

/**
 * public function from PetscDataType to stream
 * @param os
 * @param dt
 * @return
 */
std::ostream& operator<<(std::ostream& os, const PetscDataType& dt);

}  // namespace ablate::utilities
#endif  // ABLATELIBRARY_PETSCUTILITIES_HPP

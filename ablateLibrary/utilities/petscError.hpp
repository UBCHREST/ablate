#ifndef ABLATELIBRARY_PETSCERROR_HPP
#define ABLATELIBRARY_PETSCERROR_HPP
#include <petscsys.h>
#include <iostream>

namespace ablate {
namespace utilities {
class PetscErrorChecker {
   public:
    struct PetscError : public std::runtime_error {
       private:
        static std::string GetMessage(PetscErrorCode ierr) {
            const char *text;
            char *specific;

            PetscErrorMessage(ierr, &text, &specific);

            return std::string(text) + ": " + std::string(specific);
        }

       public:
        PetscError(PetscErrorCode ierr) : std::runtime_error(GetMessage(ierr)) {}
    };

    friend void operator>>(PetscErrorCode ierr, const PetscErrorChecker &errorChecker) {
        if (ierr != 0) {
            throw PetscError(ierr);
        }
    }
};
}  // namespace utilities

inline utilities::PetscErrorChecker checkError;
}  // namespace ablate

#endif  // ABLATELIBRARY_PETSCERROR_HPP

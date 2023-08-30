#ifndef ABLATELIBRARY_PETSCUTILITIES_HPP
#define ABLATELIBRARY_PETSCUTILITIES_HPP
#include <petsc.h>
#include <slepc.h>
#include <map>
#include <string>
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

    /**
     * Support for setting global petsc options with a prefix and options map
     * @param prefix
     * @param options
     * @param override force override existing options if present (default true)
     */
    static void Set(const std::string& prefix, const std::map<std::string, std::string>& options, bool override = true);

    /**
     * Support for setting global petsc options with a msp
     * @param prefix
     * @param options
     */
    static void Set(const std::map<std::string, std::string>& options);

    /**
     * Set specific petsc options object
     * @param petscOptions
     * @param options
     */
    static void Set(PetscOptions petscOptions, const std::map<std::string, std::string>& options);

    /**
     * Set specific petsc options object.  If overide is false, this will not replace exisiting options
     * @param petscOptions
     * @param name
     * @param value
     * @param override
     */
    static void Set(PetscOptions petscOptions, const char name[], const char value[], bool override = true);

    /**
     * Clean up and check for unused options
     * @param name
     * @param options
     */
    static void PetscOptionsDestroyAndCheck(const std::string& name, PetscOptions* options);

    // keep this class static
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

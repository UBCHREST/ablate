#ifndef ABLATELIBRARY_PETSCUTILITIES_HPP
#define ABLATELIBRARY_PETSCUTILITIES_HPP
#include <petsc.h>

namespace ablate::utilities {

class PetscUtilities {
   public:
    /**
     * static call to setup petsc petsc and register cleanup call
     */
    static void Initialize(const char[] = nullptr);

   private:
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

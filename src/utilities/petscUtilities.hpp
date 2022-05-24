#ifndef ABLATELIBRARY_PETSCUTILITIES_HPP
#define ABLATELIBRARY_PETSCUTILITIES_HPP

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
}  // namespace ablate::utilities
#endif  // ABLATELIBRARY_PETSCUTILITIES_HPP

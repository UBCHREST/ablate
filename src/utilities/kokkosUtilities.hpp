#ifndef ABLATELIBRARY_KOKKOSUTILITIES_HPP
#define ABLATELIBRARY_KOKKOSUTILITIES_HPP

namespace ablate::utilities {

class KokkosUtilities {
   public:
    /**
     * static call to setup petsc petsc and register cleanup call
     */
    static void Initialize();

   private:
    KokkosUtilities() = delete;
};

}  // namespace ablate::utilities
#endif  // ABLATELIBRARY_KOKKOSUTILITIES_HPP

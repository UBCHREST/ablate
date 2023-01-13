#ifndef ABLATE_LIBRARY_CONSTANTS_HPP
#define ABLATE_LIBRARY_CONSTANTS_HPP

#include <petscsystypes.h>

namespace ablate::utilities {
class Constants {
   public:
    //! Stefan-Boltzman Constant (J/K)
    constexpr inline static PetscReal sbc = 5.6696e-8;

    //! Planck Constant
    constexpr inline static PetscReal h = 6.62607004e-34;

    //! Speed of light
    constexpr inline static PetscReal c = 299792458;

    //! Boltzmann Constant
    constexpr inline static PetscReal k = 1.380622e-23;

    //! Pi
    constexpr inline static PetscReal pi = 3.1415926535897932384626433832795028841971693993;

    //! A very tiny number
    constexpr inline static PetscReal tiny = 1e-30;

    //! A somewhat small number
    constexpr inline static PetscReal small = 1e-10;

    //! A somewhat large number
    constexpr inline static PetscReal large = 1E10;
};
}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_CONSTANTS_HPP

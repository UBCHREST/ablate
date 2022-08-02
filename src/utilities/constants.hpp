#ifndef ABLATELIBRARY_CONSTANTS_HPP
#define ABLATELIBRARY_CONSTANTS_HPP

#include <petscsystypes.h>

namespace ablate::utilities {
class Constants {
   public:
    /// Class Constants
    inline static const PetscReal sbc = 5.6696e-8;  //!< Stefan-Boltzman Constant (J/K)
    inline static const PetscReal pi = 3.1415926535897932384626433832795028841971693993;
};
}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_CONSTANTS_HPP

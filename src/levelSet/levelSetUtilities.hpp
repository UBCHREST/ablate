#ifndef ABLATELIBRARY_LEVELSETUTILITIES_HPP
#define ABLATELIBRARY_LEVELSETUTILITIES_HPP

#include "LS-VOF.hpp"



namespace ablate::levelSet::Utilities {

  void VOF(DM dm, const PetscInt p, const PetscReal c0, const PetscReal n[], PetscReal *vof, PetscReal *area, PetscReal *vol);



}  // namespace ablate::levelSet::Utilities
#endif  // ABLATELIBRARY_LEVELSETUTILITIES_HPP

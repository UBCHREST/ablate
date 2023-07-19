#ifndef ABLATELIBRARY_LEVELSETGEOMETRY_HPP
#define ABLATELIBRARY_LEVELSETGEOMETRY_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "domain/RBF/rbf.hpp"


namespace ablate::levelSet::geometry {

      // Public curvature and normal functions
      PetscReal curvature(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c);
      void normal(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c, PetscReal *n);

}

#endif  // ABLATELIBRARY_LEVELSETGEOMETRY_HPP

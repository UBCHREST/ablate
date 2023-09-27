#ifndef ABLATELIBRARY_LEVELSETGEOMETRY_HPP
#define ABLATELIBRARY_LEVELSETGEOMETRY_HPP

#include <petsc.h>
#include "domain/RBF/rbf.hpp"
#include "domain/field.hpp"


namespace ablate::levelSet::geometry {

      // Public curvature and normal functions
      PetscReal Curvature(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c);
      void Normal(std::shared_ptr<ablate::domain::rbf::RBF> rbf, const ablate::domain::Field *field, PetscInt c, PetscReal *n);

}


#endif  // ABLATELIBRARY_LEVELSETGEOMETRY_HPP

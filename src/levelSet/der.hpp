#ifndef ABLATELIBRARY_DER_HPP
#define ABLATELIBRARY_DER_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "utilities/petscError.hpp"
#include "rbf.hpp"
#include "lsSupport.hpp"

namespace ablate::levelSet {

  PetscReal EvalDer(Vec f, PetscInt n, PetscInt lst[], PetscInt der, PetscInt nDer, PetscReal wt[]);
  PetscErrorCode GetDerivativeStencils(std::shared_ptr<RBF> rbf, PetscInt m, PetscInt d, PetscReal h, PetscInt **nStencilOut, PetscInt ***stencilListOut, PetscReal ***stencilWeightsOut);

}  // namespace ablate::levelSet


#endif  // ABLATELIBRARY_DER_HPP

#ifndef ABLATELIBRARY_FACESTENCILGENERATOR_HPP
#define ABLATELIBRARY_FACESTENCILGENERATOR_HPP

#include <petsc.h>
#include <vector>
#include "stencil.hpp"

namespace ablate::finiteVolume::stencil {

/**
 * generator for face based interpolants
 */
class FaceStencilGenerator {
   public:
    virtual ~FaceStencilGenerator() = default;

    virtual void Generate(PetscInt face, Stencil& stencil, const domain::SubDomain& subDomain, const std::shared_ptr<domain::Region>& solverRegion, DM cellDM, const PetscScalar* cellGeomArray,
                          DM faceDM, const PetscScalar* faceGeomArray) = 0;
};

}  // namespace ablate::finiteVolume::stencil
#endif  // ABLATELIBRARY_STENCIL_HPP

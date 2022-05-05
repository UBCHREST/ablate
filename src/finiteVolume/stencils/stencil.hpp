#ifndef ABLATELIBRARY_STENCIL_HPP
#define ABLATELIBRARY_STENCIL_HPP

#include <petsc.h>
#include <vector>

namespace ablate::finiteVolume::stencil {

/**
 * struct to hold the gradient stencil for the boundary
 */
struct Stencil {
    /** store the stencil size for easy access */
    PetscInt stencilSize = 0;
    /** The points in the stencil*/
    std::vector<PetscInt> stencil;
    /** The weights in [point*dim + dir] order */
    std::vector<PetscScalar> weights;
    /** The gradient weights in [point*dim + dir] order */
    std::vector<PetscScalar> gradientWeights;
};

}  // namespace ablate::finiteVolume::stencil
#endif  // ABLATELIBRARY_STENCIL_HPP

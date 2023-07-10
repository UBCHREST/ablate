#ifndef ABLATELIBRARY_SUBLIMATIONMODEL_HPP
#define ABLATELIBRARY_SUBLIMATIONMODEL_HPP

#include <petsc.h>
#include "boundarySolver/boundarySolver.hpp"

namespace ablate::boundarySolver::subModels {

class SublimationModel {
   public:
    /**
     * Simple struct to hold the return state of the boundary condition
     */
    struct SurfaceState {
        //! The resulting mass flux off the surface
        PetscReal massFlux;

        //! Surface temperature
        PetscReal temperature;

        //! PetscReal resulting regression rate off the surface
        PetscReal regressionRate;
    };

    /**
     * Initialize the subModel for each face id in the bSolver
     * @param bSolver
     */
    virtual BoundarySolver::BoundaryPreRHSPointFunctionDefinition Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
        return BoundarySolver::BoundaryPreRHSPointFunctionDefinition{.function = nullptr, .context = nullptr, .inputFieldsOffset = {}, .auxFieldsOffset = {}};
    }

    /**
     * Returns the current surface state for a face and current heatflux
     * @param heatFluxToSurface
     */
    virtual PetscErrorCode Solve(PetscInt faceId, PetscReal heatFluxToSurface, SurfaceState &) = 0;

    /**
     * Allow model cleanup
     */
    virtual ~SublimationModel() = default;
};
}  // namespace ablate::boundarySolver::subModels

#endif  // ABLATELIBRARY_SUBLIMATIONMODEL_HPP

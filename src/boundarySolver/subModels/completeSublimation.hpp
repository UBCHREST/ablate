#ifndef ABLATELIBRARY_COMPLETESUBLIMATION_HPP
#define ABLATELIBRARY_COMPLETESUBLIMATION_HPP

#include "sublimationModel.hpp"
namespace ablate::boundarySolver::subModels {

class CompleteSublimation : public SublimationModel {
   private:
    //! the latent heat of fusion [J/kg]"
    const PetscReal latentHeatOfFusion;

    //! Solid density of the fuel.  This is only used to output/report the solid regression rate. (Default is 1.0)
    const PetscReal solidDensity;

   public:
    explicit CompleteSublimation(PetscReal latentHeatOfFusion, PetscReal solidDensity = 1.0);

    /**
     * Returns the current surface state for a face and current heatflux
     * @param heatFluxToSurface
     */
    PetscErrorCode Solve(PetscInt faceId, PetscReal heatFluxToSurface, SurfaceState &) override;
};
}  // namespace ablate::boundarySolver::subModels

#endif  // ABLATELIBRARY_SUBLIMATIONMODEL_HPP

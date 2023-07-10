#ifndef ABLATELIBRARY_SUBLIMATIONTEMPERATURE_HPP
#define ABLATELIBRARY_SUBLIMATIONTEMPERATURE_HPP

#include <map>
#include <memory>
#include "oneDimensionHeatTransfer.hpp"
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"
#include "sublimationModel.hpp"

namespace ablate::boundarySolver::physics::subModels {

class TemperatureSublimation : public SublimationModel {
   private:
    //! hold onto a map of solid heat transfer
    std::map<PetscInt, std::shared_ptr<OneDimensionHeatTransfer>> oneDimensionHeatTransfer;

    //! hold onto a map of solid heat transfer flux, updated each time
    std::map<PetscInt, PetscReal> heatFluxIntoSolid;

    //! the material properties
    const std::shared_ptr<ablate::parameters::Parameters> properties;

    //! the math function used to initialize the domain
    const std::shared_ptr<ablate::mathFunctions::MathFunction> initialization;

    //! the petsc options used to setup each oneDimensionHeatTransfer
    const std::shared_ptr<ablate::parameters::Parameters> options;

    //! the latent heat of fusion [J/kg]"
    const PetscReal latentHeatOfFusion;

    //! Solid density of the fuel.  This is only used to output/report the solid regression rate. (Default is 1.0)
    const PetscReal solidDensity;

   public:
    TemperatureSublimation(const std::shared_ptr<ablate::parameters::Parameters> &properties, const std::shared_ptr<ablate::mathFunctions::MathFunction> &initialization,
                           const std::shared_ptr<ablate::parameters::Parameters> &options = {});

    /**
     * Initialize the subModel for each face id in the bSolver
     * @param bSolver
     */
    void Initialize(ablate::boundarySolver::BoundarySolver &bSolver) override;

    /**
     * Initialize the subModel for each face id in the bSolver
     * @param bSolver
     * @return bool indicating if this model needs to be updated before each prestep
     */
    PetscErrorCode Update(PetscInt faceId, PetscReal dt, PetscReal heatFluxToSurface, PetscReal &temperature) override;

    /**
     * Returns the current surface state for a face and current heatflux
     * @param heatFluxToSurface
     */
    PetscErrorCode Compute(PetscInt faceId, PetscReal heatFluxToSurface, SurfaceState &) override;
};

}  // namespace ablate::boundarySolver::physics::subModels
#endif  // ABLATELIBRARY_SUBLIMATIONTEMPERATURE_HPP

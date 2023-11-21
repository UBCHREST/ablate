#ifndef ABLATELIBRARY_ARRHENIUSSUBLIMATION_HPP
#define ABLATELIBRARY_ARRHENIUSSUBLIMATION_HPP

#include <map>
#include <memory>
#include "oneDimensionHeatTransfer.hpp"
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"
#include "sublimationModel.hpp"

namespace ablate::boundarySolver::physics::subModels {

class ArrheniusSublimation : public SublimationModel {
   private:
    //! hold onto a map of solid heat transfer
    std::map<PetscInt, std::shared_ptr<OneDimensionHeatTransfer>> oneDimensionHeatTransfer;

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

    //! Store the parameters needed for the reaction
    const PetscReal preExponentialFactor;
    const PetscReal activationEnergy;
    const PetscReal parameterB;

    //! universal gas constant
    constexpr inline static PetscReal ugc = 8.31446261815324;  // J/K-mol

    /**
     * Compute the current rate based upon the temperature
     * @param temperature
     * @return
     */
    [[nodiscard]] PetscReal ComputeMassFluxRate(PetscReal temperature) const;

   public:
    ArrheniusSublimation(const std::shared_ptr<ablate::parameters::Parameters> &properties, const std::shared_ptr<ablate::mathFunctions::MathFunction> &initialization,
                         const std::shared_ptr<ablate::parameters::Parameters> &options = {});

    /**
     * Initialize the subModel for each face id in the bSolver
     * @param bSolver
     */
    void Initialize(ablate::boundarySolver::BoundarySolver &bSolver) override;

    /**
     * bool indicating if this model needs to be updated before each prestep
     * @param bSolver
     * @return
     */
    bool RequiresUpdate() override { return true; };

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

    /**
     * The temperature sublimation model needs to save/restore the 1D fields
     * @return
     */
    [[nodiscard]] SerializerType Serialize() const override { return SerializerType::serial; }

    /**
     * Save the state to the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * Restore the state from the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;
};

}  // namespace ablate::boundarySolver::physics::subModels
#endif  // ABLATELIBRARY_SUBLIMATIONTEMPERATURE_HPP

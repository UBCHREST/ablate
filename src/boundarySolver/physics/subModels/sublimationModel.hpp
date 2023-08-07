#ifndef ABLATELIBRARY_SUBLIMATIONMODEL_HPP
#define ABLATELIBRARY_SUBLIMATIONMODEL_HPP

#include <petsc.h>
#include "boundarySolver/boundarySolver.hpp"

namespace ablate::boundarySolver::physics::subModels {

class SublimationModel : public io::Serializable {
   private:
    // static name of this model
    inline const static std::string sublimationModelId = "SublimationModel";

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
     * bool indicating if this model needs to be updated before each prestep
     * @param bSolver
     * @return
     */
    virtual bool RequiresUpdate() { return false; };

    /**
     * Initialize the subModel for each face id in the bSolver
     * @param bSolver
     * @return bool indicating if this model needs to be updated before each prestep
     */
    virtual void Initialize(ablate::boundarySolver::BoundarySolver &bSolver){};

    /**
     * Initialize the subModel for each face id in the bSolver
     * @param bSolver
     * @return bool indicating if this model needs to be updated before each prestep
     */
    virtual PetscErrorCode Update(PetscInt faceId, PetscReal dt, PetscReal heatFluxToSurface, PetscReal &temperature) { return PETSC_SUCCESS; };

    /**
     * Returns the current surface state for a face and current heatflux
     * @param heatFluxToSurface
     */
    virtual PetscErrorCode Compute(PetscInt faceId, PetscReal heatFluxToSurface, SurfaceState &) = 0;

    /**
     * Allow model cleanup
     */
    ~SublimationModel() override = default;

    /**
     * assume that the sublimation model does not need to Serialize
     * @return
     */
    [[nodiscard]] SerializerType Serialize() const override { return io::Serializable::SerializerType::none; }

    /**
     * only required function, returns the id of the object.  Should be unique for the simulation
     * @return
     */
    [[nodiscard]] const std::string &GetId() const override { return sublimationModelId; }

    /**
     * Save the state to the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override { return PETSC_SUCCESS; };

    /**
     * Restore the state from the PetscViewer
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override { return PETSC_SUCCESS; }
};
}  // namespace ablate::boundarySolver::physics::subModels

#endif  // ABLATELIBRARY_SUBLIMATIONMODEL_HPP

#ifndef ABLATELIBRARY_SUBLIMATION_HPP
#define ABLATELIBRARY_SUBLIMATION_HPP

#include "boundarySolver/boundaryProcess.hpp"
namespace ablate::boundarySolver::physics {

/**
 * produces required source terms in the "gas phase" assuming that the solid phase sublimates and no regression compared to the simulation time
 */
class Sublimation : public BoundaryProcess {
   private:
    const PetscReal latentHeatOfFusion;
    const PetscReal effectiveConductivity;
    const std::string offGasSpecies;
    PetscInt speciesLoc = -1;

   public:
    explicit Sublimation(PetscReal latentHeatOfFusion, PetscReal effectiveConductivity, std::string offGasSpecies = "");

    void Initialize(ablate::boundarySolver::BoundarySolver &bSolver) override;

    static PetscErrorCode SublimationFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell, const PetscInt uOff[],
                                              const PetscScalar *boundaryValues, const PetscScalar *stencilValues[], const PetscInt aOff[], const PetscScalar *auxValues,
                                              const PetscScalar *stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[],
                                              PetscScalar source[], void *ctx);
};

}  // namespace ablate::boundarySolver::physics

#endif  // ABLATELIBRARY_SUBLIMATION_HPP

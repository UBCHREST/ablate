#ifndef ABLATELIBRARY_LODI_MASSFLUXINLET_HPP
#define ABLATELIBRARY_LODI_MASSFLUXINLET_HPP

#include "lodiBoundary.hpp"

namespace ablate::boundarySolver::lodi {
// This class assumes Isothermal inlet with constant mass flux
class MassFluxInlet : public LODIBoundary {
   private:
    //! The prescribed velocity in normal cartesian components (vx, vy, vz)
    std::shared_ptr<ablate::mathFunctions::MathFunction> prescribedMassFlux;
    static PetscErrorCode UpdateMassFluxFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    explicit MassFluxInlet(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling = {},
                           std::shared_ptr<ablate::mathFunctions::MathFunction> prescribedVelocity = {});

    void Setup(ablate::boundarySolver::BoundarySolver& bSolver) override;

    static PetscErrorCode InletFunction(PetscInt dim, const boundarySolver::BoundarySolver::BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell, const PetscInt uOff[],
                                        const PetscScalar* boundaryValues, const PetscScalar* stencilValues[], const PetscInt aOff[], const PetscScalar* auxValues,
                                        const PetscScalar* stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[],
                                        PetscScalar source[], void* ctx);
};

}  // namespace ablate::boundarySolver::lodi
#endif  // ABLATELIBRARY_MASSFLUXINLET_HPP

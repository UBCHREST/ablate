#ifndef ABLATELIBRARY_ISOTHERMALWALL_HPP
#define ABLATELIBRARY_ISOTHERMALWALL_HPP

#include "lodiBoundary.hpp"
namespace ablate::boundarySolver::lodi {

class IsothermalWall : public LODIBoundary {
   public:
    explicit IsothermalWall(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling = {});

    void Initialize(ablate::boundarySolver::BoundarySolver& bSolver) override;

    static PetscErrorCode IsothermalWallFunction(PetscInt dim, const boundarySolver::BoundarySolver::BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell, const PetscInt uOff[],
                                                 const PetscScalar* boundaryValues, const PetscScalar* stencilValues[], const PetscInt aOff[], const PetscScalar* auxValues,
                                                 const PetscScalar* stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[],
                                                 PetscScalar source[], void* ctx);

   private:
    static PetscErrorCode MirrorSpecies(PetscInt dim, const BoundarySolver::BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell, const PetscInt uOff[], PetscScalar* boundaryValues,
                                        const PetscScalar* stencilValues, const PetscInt aOff[], PetscScalar* auxValues, const PetscScalar* stencilAuxValues, void* ctx);
};

}  // namespace ablate::boundarySolver::lodi
#endif  // ABLATELIBRARY_ISOTHERMALWALL_HPP

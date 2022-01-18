#ifndef ABLATELIBRARY_LODI_INLET_HPP
#define ABLATELIBRARY_LODI_INLET_HPP

#include "lodiBoundary.hpp"

namespace ablate::boundarySolver::lodi {

class Inlet : public LODIBoundary {
   public:
    explicit Inlet(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling = {});

    void Initialize(ablate::boundarySolver::BoundarySolver& bSolver) override;

    static PetscErrorCode InletFunction(PetscInt dim, const boundarySolver::BoundarySolver::BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell, const PetscInt uOff[],
                                        const PetscScalar* boundaryValues, const PetscScalar* stencilValues[], const PetscInt aOff[], const PetscScalar* auxValues,
                                        const PetscScalar* stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[],
                                        PetscScalar source[], void* ctx);
};

}  // namespace ablate::boundarySolver::lodi
#endif  // ABLATELIBRARY_INLET_HPP

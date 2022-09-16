#ifndef ABLATELIBRARY_OPENBOUNDARY_HPP
#define ABLATELIBRARY_OPENBOUNDARY_HPP

#include "lodiBoundary.hpp"
namespace ablate::boundarySolver::lodi {

class OpenBoundary : public LODIBoundary {
   private:
    // Boundary reflection factor
    const PetscReal reflectFactor;
    // Reference pressure
    const PetscReal referencePressure;
    // Max Reference Length
    const PetscReal maxAcousticsLength;

   public:
    OpenBoundary(std::shared_ptr<eos::EOS> eos, double reflectFactor, double referencePressure, double maxAcousticsLength,
                 std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling = {});

    void Setup(ablate::boundarySolver::BoundarySolver& bSolver) override;

    static PetscErrorCode OpenBoundaryFunction(PetscInt dim, const boundarySolver::BoundarySolver::BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell, const PetscInt uOff[],
                                               const PetscScalar* boundaryValues, const PetscScalar* stencilValues[], const PetscInt aOff[], const PetscScalar* auxValues,
                                               const PetscScalar* stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[],
                                               PetscScalar source[], void* ctx);
};

}  // namespace ablate::boundarySolver::lodi
#endif  // ABLATELIBRARY_OPENBOUNDARY_HPP

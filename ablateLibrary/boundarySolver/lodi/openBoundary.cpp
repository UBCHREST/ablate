#include "openBoundary.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::boundarySolver::lodi::OpenBoundary::OpenBoundary(std::shared_ptr<eos::EOS> eos, double reflectFactor, double referencePressure)
    : LODIBoundary(std::move(eos)), reflectFactor((PetscReal)reflectFactor), referencePressure((PetscReal)referencePressure) {}

void ablate::boundarySolver::lodi::OpenBoundary::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    bSolver.RegisterFunction(OpenBoundaryFunction, this, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {});
    std::cout << reflectFactor << referencePressure;
}

PetscErrorCode ablate::boundarySolver::lodi::OpenBoundary::OpenBoundaryFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell,
                                                                                const PetscInt *uOff, const PetscScalar *boundaryValues, const PetscScalar **stencilValues, const PetscInt *aOff,
                                                                                const PetscScalar *auxValues, const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                                const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    return 0;
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::lodi::OpenBoundary, "Treats boundary as open.",
         ARG(ablate::eos::EOS, "eos", "The EOS describing the flow field at the boundary"), ARG(double, "reflectFactor", "boundary reflection factor"),
         ARG(double, "referencePressure", "reference pressure"));

#include "isothermalWall.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::boundarySolver::lodi::IsothermalWall::IsothermalWall(std::shared_ptr<eos::EOS> eos) : LODIBoundary(std::move(eos)) {}
PetscErrorCode ablate::boundarySolver::lodi::IsothermalWall::IsothermalWallIsothermalWallFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg,
                                                                                                  const PetscFVCellGeom *boundaryCell, const PetscInt *uOff, const PetscScalar *boundaryValues,
                                                                                                  const PetscScalar **stencilValues, const PetscInt *aOff, const PetscScalar *auxValues,
                                                                                                  const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                                                  const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;

//    auto isothermalWall = (IsothermalWall*)ctx;

    PetscFunctionReturn(0);

}
void ablate::boundarySolver::lodi::IsothermalWall::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    bSolver.RegisterFunction(IsothermalWallIsothermalWallFunction, this, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {});
}

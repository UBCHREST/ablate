#include "isothermalWall.hpp"
#include <utilities/mathUtilities.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/flowProcess.hpp"

using fp = ablate::finiteVolume::processes::FlowProcess;

ablate::boundarySolver::lodi::IsothermalWall::IsothermalWall(std::shared_ptr<eos::EOS> eos) : LODIBoundary(std::move(eos)) {}
PetscErrorCode ablate::boundarySolver::lodi::IsothermalWall::IsothermalWallIsothermalWallFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg,
                                                                                                  const PetscFVCellGeom *boundaryCell, const PetscInt *uOff, const PetscScalar *boundaryValues,
                                                                                                  const PetscScalar **stencilValues, const PetscInt *aOff, const PetscScalar *auxValues,
                                                                                                  const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                                                  const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    /* const int EULER = 0;
    PetscErrorCode ierr;
    auto isothermalWall = (IsothermalWall *)ctx;
    auto decodeStateFunction = isothermalWall->eos->GetDecodeStateFunction();
    auto decodeStateContext = isothermalWall->eos->GetDecodeStateContext();

    // Compute the pressure at each cell
    PetscReal boundaryDensity;
    std::vector<PetscReal> boundaryVel(dim);
    PetscReal boundaryNormalVelocity;
    PetscReal boundaryInternalEnergy;
    PetscReal boundarySpeedOfSound;
    PetscReal boundaryMach;
    PetscReal boundaryPressure;

    // Get the velocity and pressure on the surface
    finiteVolume::processes::FlowProcess::DecodeEulerState(decodeStateFunction,
                                                           decodeStateContext,
                                                           dim,
                                                           boundaryValues + uOff[0],
                                                           nullptr,
                                                           fg->normal,
                                                           &boundaryDensity,
                                                           &boundaryNormalVelocity,
                                                           &boundaryVel[0],
                                                           &boundaryInternalEnergy,
                                                           &boundarySpeedOfSound,
                                                           &boundaryMach,
                                                           &boundaryPressure);

    // Compute each stencil point
    std::vector<PetscReal> stencilDensity(stencilSize);
    std::vector<std::vector<PetscReal>> stencilVel(stencilSize, std::vector<PetscReal>(dim));
    std::vector<PetscReal> stencilInternalEnergy(stencilSize);
    std::vector<PetscReal> stencilNormalVelocity(stencilSize);
    std::vector<PetscReal> stencilSpeedOfSound(stencilSize);
    std::vector<PetscReal> stencilMach(stencilSize);
    std::vector<PetscReal> stencilPressure(stencilSize);

    for (PetscInt s = 0; s < stencilSize; s++) {
        finiteVolume::processes::FlowProcess::DecodeEulerState(decodeStateFunction,
                                                               decodeStateContext,
                                                               dim,
                                                               stencilValues[s] + uOff[0],
                                                               nullptr,
                                                               fg->normal,
                                                               &stencilDensity[s],
                                                               &stencilNormalVelocity[s],
                                                               &stencilVel[s][0],
                                                               &stencilInternalEnergy[s],
                                                               &stencilSpeedOfSound[s],
                                                               &stencilMach[s],
                                                               &stencilPressure[s]);
    }

    // Interpolate the normal velocity gradient to the surface
    PetscReal normalVelocityGradientCard[3] = {NAN, NAN, NAN};
    BoundarySolver::ComputeGradient(dim, boundaryNormalVelocity, stencilSize, &stencilNormalVelocity[0], stencilWeights, normalVelocityGradientCard);

    // Map to normal coord
    PetscScalar transformationMatrix[3][3];
    utilities::MathUtilities::ComputeTransformationMatrix(dim, fg->normal, transformationMatrix);
    PetscScalar dVeldNorm[3];
    utilities::MathUtilities::Multiply(dim, transformationMatrix, normalVelocityGradientCard, dVeldNorm);

    // Compute the cp, cv from the eos
*/

    PetscFunctionReturn(0);
}
void ablate::boundarySolver::lodi::IsothermalWall::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    bSolver.RegisterFunction(IsothermalWallIsothermalWallFunction, this, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {});
}

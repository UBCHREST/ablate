#include "logLawBoundary.hpp"
#include <utilities/mathUtilities.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"

void ablate::boundarySolver::physics::LogLawBoundary::Setup(ablate::boundarySolver::BoundarySolver &bSolver) {
    bSolver.RegisterFunction(UpdateBoundaryVel, this, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {finiteVolume::CompressibleFlowFields::VELOCITY_FIELD});
}

PetscErrorCode ablate::boundarySolver::physics::LogLawBoundary::UpdateBoundaryVel(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg,
                                                                                  const PetscFVCellGeom *boundaryCell, const PetscInt *uOff, PetscScalar *boundaryValues,
                                                                                  const PetscScalar *stencilValues, const PetscInt *aOff, PetscScalar *auxValues, const PetscScalar *stencilAuxValues,
                                                                                  void *ctx) {
    PetscFunctionBeginUser;
    const PetscInt EULER_FIELD = 0;
    const PetscInt VEL = 0;

    PetscReal logLawVel[3];
    PetscArrayzero(logLawVel, 3);
    PetscReal stencilVel[3];
    PetscReal stencilNormalVelocity[3];
    PetscReal tangVel_1;
    PetscReal tangVel_2;

    // map the stencil velocity in normal coord.
    PetscReal stencilDensity = stencilValues[uOff[EULER_FIELD] + finiteVolume::CompressibleFlowFields::RHO];
    for (PetscInt d = 0; d < dim; d++) {
        stencilVel[d] = stencilValues[uOff[EULER_FIELD] + finiteVolume::CompressibleFlowFields::RHOU + d] / stencilDensity;
    }

    PetscReal transformationMatrix[dim][3];
    utilities::MathUtilities::ComputeTransformationMatrix(dim, fg->normal, transformationMatrix);
    ablate::utilities::MathUtilities::Multiply(dim, transformationMatrix, stencilVel, stencilNormalVelocity);
    PetscReal area = utilities::MathUtilities::MagVector(dim, fg->areas);

    // calculate the new boundary velocity in normal coord for 2D.
    logLawVel[0] = 0e+0;

    logLawVel[1] =
        -fg->normal[1] * 0.5 * (1 / kappa) * 2 / (boundaryCell->volume * boundaryCell->volume / area * area) / (4 / (boundaryCell->volume * boundaryCell->volume / area * area) - 4 / (area)) +
        0.5 * stencilNormalVelocity[1];

    // for 3D
    if (dim == 3) {
        if (abs(stencilNormalVelocity[2]) >= abs(stencilNormalVelocity[1])) {
            tangVel_1 =
                fg->normal[1] * 0.5 * (1 / kappa) * 2 / (boundaryCell->volume * boundaryCell->volume / area * area) / (4 / (boundaryCell->volume * boundaryCell->volume / area * area) - 4 / (area)) +
                0.5 * stencilNormalVelocity[2];

            tangVel_2 = stencilNormalVelocity[1];

        } else {
            tangVel_1 =
                fg->normal[1] * 0.5 * (1 / kappa) * 2 / (boundaryCell->volume * boundaryCell->volume / area * area) / (4 / (boundaryCell->volume * boundaryCell->volume / area * area) - 4 / (area)) +
                0.5 * stencilNormalVelocity[1];

            tangVel_2 = stencilNormalVelocity[2];
        }

        PetscReal newMagTangVel = sqrt(tangVel_1 * tangVel_1 + tangVel_2 * tangVel_2);
        PetscReal magTangVel = sqrt(stencilNormalVelocity[1] * stencilNormalVelocity[1] + stencilNormalVelocity[2] * stencilNormalVelocity[2]);

        if (magTangVel == 0) {
            logLawVel[1] = tangVel_1;
            logLawVel[2] = tangVel_2;

        } else {
            logLawVel[1] = (newMagTangVel / magTangVel) * stencilNormalVelocity[1];
            logLawVel[2] = (newMagTangVel / magTangVel) * stencilNormalVelocity[2];
        }
    }
    // map the boundary velocities back into Cartesian coord.
    PetscReal boundaryVel[dim];
    PetscReal velocityCartSystem[dim];

    ablate::utilities::MathUtilities::MultiplyTranspose(dim, transformationMatrix, logLawVel, velocityCartSystem);
    // update boundary velocities
    PetscScalar boundaryDensity = boundaryValues[uOff[EULER_FIELD] + finiteVolume::CompressibleFlowFields::RHO];
    for (PetscInt d = 0; d < dim; d++) {
        boundaryVel[d] = velocityCartSystem[d];
        boundaryValues[uOff[EULER_FIELD] + finiteVolume::CompressibleFlowFields::RHOU + d] = boundaryVel[d] * boundaryDensity;
        auxValues[aOff[VEL] + d] = boundaryVel[d];
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::physics::LogLawBoundary,
                           "updates boundary velocities at the wall using outflow to reconstruct the log law");
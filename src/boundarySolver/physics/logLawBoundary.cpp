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
    PetscReal logLawVel[dim];
    PetscArrayzero(logLawVel, 3);
    PetscReal stencilVel[3];
    PetscReal stencilNormalVelocity[3];

    // map the stencil velocity in normal coord.
    PetscReal stencilDensity = stencilValues[uOff[EULER_FIELD] + finiteVolume::CompressibleFlowFields::RHO];
    for (PetscInt d = 0; d < dim; d++) {
        stencilVel[d] = stencilValues[uOff[EULER_FIELD] + finiteVolume::CompressibleFlowFields::RHOU + d] / stencilDensity;
    }

    PetscReal transformationMatrix[dim][3];
    utilities::MathUtilities::ComputeTransformationMatrix(dim, fg->normal, transformationMatrix);
    ablate::utilities::MathUtilities::Multiply(dim, transformationMatrix, stencilVel, stencilNormalVelocity);

    // calculate the new boundary velocity in normal coord.
    logLawVel[1] = -fg->normal[1] * -0.5 * (1 / kappa) * 2 / ((boundaryCell->volume * boundaryCell->volume / (fg->areas[1] * fg->areas[1])) * abs(fg->normal[1])) /
                       (4 / ((boundaryCell->volume * boundaryCell->volume / (fg->areas[1] * fg->areas[1])) * abs(fg->normal[1])) - 4 / (fg->areas[1])) -
                   fg->normal[1] * 0.5 * stencilNormalVelocity[1];

    logLawVel[0] = 0e+0;
    logLawVel[2] = stencilNormalVelocity[2];

    // map the boundary velocitiy into Cartesian coord.
    PetscReal boundaryVel[3];
    PetscReal velocityCartSystem[3];

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
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
    PetscReal theta;
    PetscReal tangVel;
    PetscReal tangBoundaryVel;

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
        // map tangential velocities into the flow direction such that there is one velocity vector
        theta = atan(stencilNormalVelocity[2] / stencilNormalVelocity[1]);
        tangVel = stencilNormalVelocity[1] * cos(theta) + stencilNormalVelocity[2] * sin(theta);

        // calculate the tangential boundary velocity
        tangBoundaryVel =
            fg->normal[1] * 0.5 * (1 / kappa) * 2 / (boundaryCell->volume * boundaryCell->volume / area * area) / (4 / (boundaryCell->volume * boundaryCell->volume / area * area) - 4 / (area)) +
            0.5 * tangVel;

        // map the tangential boundary velocity into original normal coordinate system
        logLawVel[1] = tangBoundaryVel * cos(theta);
        logLawVel[2] = tangBoundaryVel * sin(theta);
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
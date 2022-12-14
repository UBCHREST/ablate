#include "inlet.hpp"
#include <utilities/mathUtilities.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "mathFunctions/functionFactory.hpp"

using fp = ablate::finiteVolume::CompressibleFlowFields;

ablate::boundarySolver::lodi::Inlet::Inlet(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling,
                                           std::shared_ptr<ablate::mathFunctions::MathFunction> prescribedVelocity)
    : LODIBoundary(std::move(eos), std::move(pressureGradientScaling)), prescribedVelocity(std::move(prescribedVelocity)) {}

void ablate::boundarySolver::lodi::Inlet::Setup(ablate::boundarySolver::BoundarySolver &bSolver) {
    ablate::boundarySolver::lodi::LODIBoundary::Setup(bSolver);

    bSolver.RegisterFunction(InletFunction, this, fieldNames, fieldNames, {});

    // Register a pre function step to update velocity over this solver if specified
    if (prescribedVelocity) {
        // define an update field function
        auto updateFieldFunction =
            std::make_shared<mathFunctions::FieldFunction>(finiteVolume::CompressibleFlowFields::EULER_FIELD, ablate::mathFunctions::Create(UpdateVelocityFunction, prescribedVelocity.get()));

        bSolver.RegisterPreStep([&bSolver, updateFieldFunction](auto ts, auto &solver) {
            // Get the current time
            PetscReal time;
            TSGetTime(ts, &time) >> checkError;

            bSolver.InsertFieldFunctions({updateFieldFunction}, time);
        });
    }
}
PetscErrorCode ablate::boundarySolver::lodi::Inlet::InletFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell,
                                                                  const PetscInt *uOff, const PetscScalar *boundaryValues, const PetscScalar **stencilValues, const PetscInt *aOff,
                                                                  const PetscScalar *auxValues, const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                  const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    auto inletBoundary = (Inlet *)ctx;

    // Compute the transformation matrix
    PetscReal transformationMatrix[3][3];
    utilities::MathUtilities::ComputeTransformationMatrix(dim, fg->normal, transformationMatrix);

    // Compute the pressure/values on the boundary
    PetscReal boundaryDensity;
    PetscReal boundaryTemperature;
    PetscReal boundaryVel[3];
    PetscReal boundaryNormalVelocity = 0.0;
    PetscReal boundarySpeedOfSound;
    PetscReal boundaryPressure;

    // Get the densityYi pointer if available
    const PetscScalar *boundaryDensityYi = inletBoundary->nSpecEqs > 0 ? boundaryValues + uOff[inletBoundary->speciesId] : nullptr;

    // Get the velocity and pressure on the surface
    {
        boundaryDensity = boundaryValues[uOff[inletBoundary->eulerId] + finiteVolume::CompressibleFlowFields::RHO];
        for (PetscInt d = 0; d < dim; d++) {
            boundaryVel[d] = boundaryValues[uOff[inletBoundary->eulerId] + finiteVolume::CompressibleFlowFields::RHOU + d] / boundaryDensity;
            boundaryNormalVelocity += boundaryVel[d] * fg->normal[d];
        }
        PetscErrorCode ierr = inletBoundary->computeTemperature.function(boundaryValues, &boundaryTemperature, inletBoundary->computeTemperature.context.get());
        CHKERRQ(ierr);
        ierr = inletBoundary->computeSpeedOfSound.function(boundaryValues, boundaryTemperature, &boundarySpeedOfSound, inletBoundary->computeSpeedOfSound.context.get());
        CHKERRQ(ierr);
        ierr = inletBoundary->computePressureFromTemperature.function(boundaryValues, boundaryTemperature, &boundaryPressure, inletBoundary->computePressureFromTemperature.context.get());
        CHKERRQ(ierr);
    }

    // Map the boundary velocity into the normal coord system
    PetscReal boundaryVelNormCord[3];
    utilities::MathUtilities::Multiply(dim, transformationMatrix, boundaryVel, boundaryVelNormCord);

    // Compute each stencil point
    std::vector<PetscReal> stencilDensity(stencilSize);
    std::vector<std::vector<PetscReal>> stencilVel(stencilSize, std::vector<PetscReal>(dim));
    std::vector<PetscReal> stencilNormalVelocity(stencilSize);
    std::vector<PetscReal> stencilPressure(stencilSize);

    for (PetscInt s = 0; s < stencilSize; s++) {
        stencilDensity[s] = stencilValues[s][uOff[inletBoundary->eulerId] + finiteVolume::CompressibleFlowFields::RHO];
        for (PetscInt d = 0; d < dim; d++) {
            stencilVel[s][d] = stencilValues[s][uOff[inletBoundary->eulerId] + finiteVolume::CompressibleFlowFields::RHOU + d] / stencilDensity[s];
            stencilNormalVelocity[s] += stencilVel[s][d] * fg->normal[d];
        }
        PetscErrorCode ierr = inletBoundary->computePressure.function(stencilValues[s], &stencilPressure[s], inletBoundary->computePressure.context.get());
        CHKERRQ(ierr);
    }

    // Interpolate the normal velocity gradient to the surface
    PetscScalar dVeldNorm;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryNormalVelocity, stencilSize, &stencilNormalVelocity[0], stencilWeights, dVeldNorm);
    PetscScalar dPdNorm;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryPressure, stencilSize, &stencilPressure[0], stencilWeights, dPdNorm);

    // Compute the cp, cv from the eos
    std::vector<PetscReal> boundaryYi(inletBoundary->nSpecEqs);
    for (PetscInt i = 0; i < inletBoundary->nSpecEqs; i++) {
        boundaryYi[i] = boundaryDensityYi[i] / boundaryDensity;
    }
    PetscReal boundaryCp, boundaryCv;
    inletBoundary->computeSpecificHeatConstantPressure.function(boundaryValues, boundaryTemperature, &boundaryCp, inletBoundary->computeSpecificHeatConstantPressure.context.get());
    inletBoundary->computeSpecificHeatConstantVolume.function(boundaryValues, boundaryTemperature, &boundaryCv, inletBoundary->computeSpecificHeatConstantVolume.context.get());

    // Compute the enthalpy
    PetscReal boundarySensibleEnthalpy;
    inletBoundary->computeSensibleEnthalpyFunction.function(boundaryValues, boundaryTemperature, &boundarySensibleEnthalpy, inletBoundary->computeSensibleEnthalpyFunction.context.get());

    // get_vel_and_c_prims(PGS, velwall[0], C, Cp, Cv, velnprm, Cprm);
    PetscReal velNormPrim, speedOfSoundPrim;
    inletBoundary->GetVelAndCPrims(boundaryNormalVelocity, boundarySpeedOfSound, boundaryCp, boundaryCv, velNormPrim, speedOfSoundPrim);

    // get_eigenvalues
    std::vector<PetscReal> lambda(inletBoundary->nEqs);
    inletBoundary->GetEigenValues(boundaryNormalVelocity, boundarySpeedOfSound, velNormPrim, speedOfSoundPrim, &lambda[0]);

    // Get alpha
    PetscReal pgsAlpha = inletBoundary->pressureGradientScaling ? inletBoundary->pressureGradientScaling->GetAlpha() : 1.0;

    // Get scriptL
    std::vector<PetscReal> scriptL(inletBoundary->nEqs);
    // Outgoing acoustic wave
    scriptL[1 + dim] = lambda[1 + dim] * (dPdNorm - boundaryDensity * PetscSqr(pgsAlpha) * dVeldNorm * (velNormPrim - boundaryNormalVelocity - speedOfSoundPrim));

    // Incoming acoustic wave
    scriptL[0] = scriptL[1 + dim];

    // Entropy wave
    scriptL[1] = 0.5e+0 * (boundaryCp / boundaryCv - 1.e+0) * (scriptL[1 + dim] + scriptL[0]) -
                 0.5 * (boundaryCp / boundaryCv + 1.e+0) * (scriptL[0] - scriptL[1 + dim]) * (velNormPrim - boundaryNormalVelocity) / speedOfSoundPrim;

    // Tangential velocities
    for (int d = 1; d < dim; d++) {
        scriptL[1 + d] = 0.e+0;
    }
    for (int ns = 0; ns < inletBoundary->nSpecEqs; ns++) {
        // Species
        scriptL[2 + dim + ns] = 0.e+0;
    }
    for (int ne = 0; ne < inletBoundary->nEvEqs; ne++) {
        // Extra variables
        scriptL[2 + dim + inletBoundary->nSpecEqs + ne] = 0.e+0;
    }

    // Directly compute the source terms, note that this may be problem in the future with multiple source terms on the same boundary cell
    inletBoundary->GetmdFdn(sOff,
                            boundaryVelNormCord,
                            boundaryDensity,
                            boundaryTemperature,
                            boundaryCp,
                            boundaryCv,
                            boundarySpeedOfSound,
                            boundarySensibleEnthalpy,
                            velNormPrim,
                            speedOfSoundPrim,
                            boundaryValues,
                            uOff,
                            scriptL.data(),
                            transformationMatrix,
                            source);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::boundarySolver::lodi::Inlet::UpdateVelocityFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;

    auto velocityFunction = (ablate::mathFunctions::MathFunction *)ctx;

    // Get the current velocity
    PetscScalar velocity[3];
    PetscScalar kineticEnergy = 0.0;
    PetscScalar density = u[fp::RHO];
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = u[fp::RHOU + d] / density;
        kineticEnergy += PetscSqr(velocity[d]);
    }
    kineticEnergy *= 0.5;

    // Get the internal energy
    PetscScalar internalEnergy = u[fp::RHOE] / density;

    // Compute the sensible energy
    PetscScalar sensibleEnergy = internalEnergy - kineticEnergy;

    // Update velocity
    velocityFunction->GetPetscFunction()(dim, time, x, dim, velocity, velocityFunction->GetContext()) >> checkError;

    // Update the momentum terms
    kineticEnergy = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        u[fp::RHOU + d] = velocity[d] * density;
        kineticEnergy += PetscSqr(velocity[d]);
    }
    kineticEnergy *= 0.5;

    // Update the new internal energy
    u[fp::RHOE] = (sensibleEnergy + kineticEnergy) * density;

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::lodi::Inlet, "Enforces an inlet with specified velocity",
         ARG(ablate::eos::EOS, "eos", "The EOS describing the flow field at the wall"),
         OPT(ablate::finiteVolume::processes::PressureGradientScaling, "pgs", "Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows"),
         OPT(ablate::mathFunctions::MathFunction, "velocity", "optional velocity function that can change over time"));

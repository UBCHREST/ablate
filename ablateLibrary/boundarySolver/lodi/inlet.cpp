#include "inlet.hpp"
#include <utilities/mathUtilities.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"

using fp = ablate::finiteVolume::processes::FlowProcess;

ablate::boundarySolver::lodi::Inlet::Inlet(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling)
    : LODIBoundary(std::move(eos), std::move(pressureGradientScaling)) {}

void ablate::boundarySolver::lodi::Inlet::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    ablate::boundarySolver::lodi::LODIBoundary::Initialize(bSolver);

    bSolver.RegisterFunction(InletFunction, this, fieldNames, fieldNames, {});
}
PetscErrorCode ablate::boundarySolver::lodi::Inlet::InletFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell,
                                                                  const PetscInt *uOff, const PetscScalar *boundaryValues, const PetscScalar **stencilValues, const PetscInt *aOff,
                                                                  const PetscScalar *auxValues, const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                  const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    auto inletBoundary = (Inlet *)ctx;
    auto decodeStateFunction = inletBoundary->eos->GetDecodeStateFunction();
    auto decodeStateContext = inletBoundary->eos->GetDecodeStateContext();

    // Compute the transformation matrix
    PetscReal transformationMatrix[3][3];
    utilities::MathUtilities::ComputeTransformationMatrix(dim, fg->normal, transformationMatrix);

    // Compute the pressure/values on the boundary
    PetscReal boundaryDensity;
    PetscReal boundaryVel[3];
    PetscReal boundaryNormalVelocity;
    PetscReal boundaryInternalEnergy;
    PetscReal boundarySpeedOfSound;
    PetscReal boundaryMach;
    PetscReal boundaryPressure;

    // Get the densityYi pointer if available
    const PetscScalar *boundaryDensityYi = inletBoundary->nSpecEqs > 0 ? boundaryValues + uOff[inletBoundary->speciesId] : nullptr;

    // Get the velocity and pressure on the surface
    finiteVolume::processes::FlowProcess::DecodeEulerState(decodeStateFunction,
                                                           decodeStateContext,
                                                           dim,
                                                           boundaryValues + uOff[inletBoundary->eulerId],
                                                           boundaryDensityYi,
                                                           fg->normal,
                                                           &boundaryDensity,
                                                           &boundaryNormalVelocity,
                                                           boundaryVel,
                                                           &boundaryInternalEnergy,
                                                           &boundarySpeedOfSound,
                                                           &boundaryMach,
                                                           &boundaryPressure);

    // Map the boundary velocity into the normal coord system
    PetscReal boundaryVelNormCord[3];
    utilities::MathUtilities::Multiply(dim, transformationMatrix, boundaryVel, boundaryVelNormCord);

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
                                                               &stencilValues[s][uOff[inletBoundary->eulerId]],
                                                               inletBoundary->nSpecEqs > 0 ? &stencilValues[s][uOff[inletBoundary->speciesId]] : nullptr,
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
    PetscScalar dVeldNorm;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryNormalVelocity, stencilSize, &stencilNormalVelocity[0], stencilWeights, dVeldNorm);
    PetscScalar dPdNorm;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryPressure, stencilSize, &stencilPressure[0], stencilWeights, dPdNorm);

    // Compute the temperature at the boundary
    PetscReal boundaryTemperature;
    inletBoundary->eos->GetComputeTemperatureFunction()(dim,
                                                        boundaryDensity,
                                                        boundaryValues[uOff[inletBoundary->eulerId] + fp::RHOE] / boundaryDensity,
                                                        boundaryValues + uOff[inletBoundary->eulerId] + fp::RHOU,
                                                        boundaryDensityYi,
                                                        &boundaryTemperature,
                                                        inletBoundary->eos->GetComputeTemperatureContext()) >>
        checkError;

    // Compute the cp, cv from the eos
    std::vector<PetscReal> boundaryYi(inletBoundary->nSpecEqs);
    for (PetscInt i = 0; i < inletBoundary->nSpecEqs; i++) {
        boundaryYi[i] = boundaryDensityYi[i] / boundaryDensity;
    }
    PetscReal boundaryCp, boundaryCv;
    inletBoundary->eos->GetComputeSpecificHeatConstantPressureFunction()(
        boundaryTemperature, boundaryDensity, boundaryYi.data(), &boundaryCp, inletBoundary->eos->GetComputeSpecificHeatConstantPressureContext()) >>
        checkError;
    inletBoundary->eos->GetComputeSpecificHeatConstantVolumeFunction()(
        boundaryTemperature, boundaryDensity, boundaryYi.data(), &boundaryCv, inletBoundary->eos->GetComputeSpecificHeatConstantVolumeContext()) >>
        checkError;

    // Compute the enthalpy
    PetscReal boundarySensibleEnthalpy;
    inletBoundary->eos->GetComputeSensibleEnthalpyFunction()(
        boundaryTemperature, boundaryDensity, boundaryYi.data(), &boundarySensibleEnthalpy, inletBoundary->eos->GetComputeSensibleEnthalpyContext()) >>
        checkError;

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
                            boundaryDensityYi /* PetscReal* Yi*/,
                            inletBoundary->nEvEqs > 0 ? boundaryValues + uOff[inletBoundary->evId] : nullptr /* PetscReal* EV*/,
                            &scriptL[0],
                            transformationMatrix,
                            source);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::lodi::Inlet, "Enforces an inlet with specified velocity",
         ARG(ablate::eos::EOS, "eos", "The EOS describing the flow field at the wall"),
         OPT(ablate::finiteVolume::processes::PressureGradientScaling, "pgs", "Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows"));

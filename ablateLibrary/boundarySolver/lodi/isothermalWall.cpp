#include "isothermalWall.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "utilities/mathUtilities.hpp"

using fp = ablate::finiteVolume::processes::FlowProcess;

ablate::boundarySolver::lodi::IsothermalWall::IsothermalWall(std::shared_ptr<eos::EOS> eos, std::shared_ptr<finiteVolume::resources::PressureGradientScaling> pressureGradientScaling)
    : LODIBoundary(std::move(eos), std::move(pressureGradientScaling)) {}

void ablate::boundarySolver::lodi::IsothermalWall::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    ablate::boundarySolver::lodi::LODIBoundary::Initialize(bSolver);
    bSolver.RegisterFunction(IsothermalWallFunction, this, fieldNames, fieldNames, {});
}

PetscErrorCode ablate::boundarySolver::lodi::IsothermalWall::IsothermalWallFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg,
                                                                                    const PetscFVCellGeom *boundaryCell, const PetscInt uOff[], const PetscScalar *boundaryValues,
                                                                                    const PetscScalar *stencilValues[], const PetscInt aOff[], const PetscScalar *auxValues,
                                                                                    const PetscScalar *stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[],
                                                                                    const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[], void *ctx) {
    PetscFunctionBeginUser;
    auto isothermalWall = (IsothermalWall *)ctx;
    auto decodeStateFunction = isothermalWall->eos->GetDecodeStateFunction();
    auto decodeStateContext = isothermalWall->eos->GetDecodeStateContext();

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
    const PetscScalar *boundaryDensityYi = isothermalWall->nSpecEqs > 0 ? boundaryValues + uOff[isothermalWall->speciesId] : nullptr;

    // Get the velocity and pressure on the surface
    finiteVolume::processes::FlowProcess::DecodeEulerState(decodeStateFunction,
                                                           decodeStateContext,
                                                           dim,
                                                           boundaryValues + uOff[isothermalWall->eulerId],
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
                                                               &stencilValues[s][uOff[isothermalWall->eulerId]],
                                                               isothermalWall->nSpecEqs > 0 ? &stencilValues[s][uOff[isothermalWall->speciesId]] : nullptr,
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
    isothermalWall->eos->GetComputeTemperatureFunction()(dim,
                                                         boundaryDensity,
                                                         boundaryValues[uOff[isothermalWall->eulerId] + fp::RHOE] / boundaryDensity,
                                                         boundaryValues + uOff[isothermalWall->eulerId] + fp::RHOU,
                                                         boundaryDensityYi,
                                                         &boundaryTemperature,
                                                         isothermalWall->eos->GetComputeTemperatureContext()) >>
        checkError;

    // Compute the cp, cv from the eos
    std::vector<PetscReal> boundaryYi(isothermalWall->nSpecEqs);
    for (PetscInt i = 0; i < isothermalWall->nSpecEqs; i++) {
        boundaryYi[i] = boundaryDensityYi[i] / boundaryDensity;
    }

    PetscReal boundaryCp, boundaryCv;
    isothermalWall->eos->GetComputeSpecificHeatConstantPressureFunction()(
        boundaryTemperature, boundaryDensity, boundaryYi.data(), &boundaryCp, isothermalWall->eos->GetComputeSpecificHeatConstantPressureContext()) >>
        checkError;
    isothermalWall->eos->GetComputeSpecificHeatConstantVolumeFunction()(
        boundaryTemperature, boundaryDensity, boundaryYi.data(), &boundaryCv, isothermalWall->eos->GetComputeSpecificHeatConstantVolumeContext()) >>
        checkError;

    // Compute the enthalpy
    PetscReal boundarySensibleEnthalpy;
    isothermalWall->eos->GetComputeSensibleEnthalpyFunction()(
        boundaryTemperature, boundaryDensity, boundaryYi.data(), &boundarySensibleEnthalpy, isothermalWall->eos->GetComputeSensibleEnthalpyContext()) >>
        checkError;

    // get_vel_and_c_prims(PGS, velwall[0], C, Cp, Cv, velnprm, Cprm);
    PetscReal velNormPrim, speedOfSoundPrim;
    isothermalWall->GetVelAndCPrims(boundaryNormalVelocity, boundarySpeedOfSound, boundaryCp, boundaryCv, velNormPrim, speedOfSoundPrim);

    // get_eigenvalues
    std::vector<PetscReal> lambda(isothermalWall->nEqs);
    isothermalWall->GetEigenValues(boundaryNormalVelocity, boundarySpeedOfSound, velNormPrim, speedOfSoundPrim, &lambda[0]);

    // Compute alpha2
    PetscReal alpha2 = 1.0;
    if (isothermalWall->pressureGradientScaling) {
        alpha2 = PetscSqr(isothermalWall->pressureGradientScaling->GetAlpha());
    }

    // Get scriptL
    std::vector<PetscReal> scriptL(isothermalWall->nEqs);
    scriptL[1 + dim] = lambda[1 + dim] * (dPdNorm - boundaryDensity * dVeldNorm * alpha2 * (velNormPrim - boundaryNormalVelocity - speedOfSoundPrim));  // Outgoing
    // acoustic
    // wave
    scriptL[0] = scriptL[1 + dim];  // Incoming acoustic wave
    // sL[1][n1][n0] = 0.; // Entropy wave
    scriptL[1] = 0.5e+0 * (boundaryCp / boundaryCv - 1.e+0) * (scriptL[1 + dim] + scriptL[0]) -
                 (boundaryCp / boundaryCv + 1.e+0) * (scriptL[0] - scriptL[1 + dim]) * (velNormPrim - boundaryNormalVelocity) / speedOfSoundPrim;  // Entropy wave
    for (int d = 1; d < dim; d++) {
        scriptL[1 + d] = 0.e+0;  // Tangential velocities
    }
    // Species
    for (int ns = 0; ns < isothermalWall->nSpecEqs; ns++) {
        scriptL[2 + dim + ns] = 0.e+0;
    }
    // Extra variables
    for (int ne = 0; ne < isothermalWall->nEvEqs; ne++) {
        scriptL[2 + dim + isothermalWall->nSpecEqs + ne] = 0.e+0;
    }

    // Directly compute the source terms, note that this may be problem in the future with multiple source terms on the same boundary cell
    isothermalWall->GetmdFdn(sOff,
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
                             isothermalWall->nEvEqs > 0 ? boundaryValues + uOff[isothermalWall->evId] : nullptr /* PetscReal* EV*/,
                             scriptL.data(),
                             transformationMatrix,
                             source);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::lodi::IsothermalWall, "Enforces a isothermal wall with fixed velocity/temperature",
         ARG(ablate::eos::EOS, "eos", "The EOS describing the flow field at the wall"),
         OPT(ablate::finiteVolume::resources::PressureGradientScaling, "pgs", "Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows"));

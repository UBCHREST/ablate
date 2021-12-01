#include "inlet.hpp"
#include <utilities/mathUtilities.hpp>
#include "finiteVolume/compressibleFlowFields.hpp"

using fp = ablate::finiteVolume::processes::FlowProcess;

ablate::boundarySolver::lodi::Inlet::Inlet(std::shared_ptr<eos::EOS> eos)
    : LODIBoundary(std::move(eos)){}

void ablate::boundarySolver::lodi::Inlet::Initialize(ablate::boundarySolver::BoundarySolver& bSolver) {
    bSolver.RegisterFunction(InletFunction, this, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {finiteVolume::CompressibleFlowFields::EULER_FIELD}, {});
}
PetscErrorCode ablate::boundarySolver::lodi::Inlet::InletFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell,
                                                                  const PetscInt *uOff, const PetscScalar *boundaryValues, const PetscScalar **stencilValues, const PetscInt *aOff,
                                                                  const PetscScalar *auxValues, const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                  const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    const int EULER = 0;
    auto inletBoundary = (Inlet *)ctx;
    auto decodeStateFunction = inletBoundary->eos->GetDecodeStateFunction();
    auto decodeStateContext = inletBoundary->eos->GetDecodeStateContext();
    const int neq = 2 + dim;

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

    // Get the velocity and pressure on the surface
    finiteVolume::processes::FlowProcess::DecodeEulerState(decodeStateFunction,
                                                           decodeStateContext,
                                                           dim,
                                                           boundaryValues + uOff[EULER],
                                                           nullptr,
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
                                                               &stencilValues[s][uOff[EULER]],
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
    PetscScalar dVeldNorm;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryNormalVelocity, stencilSize, &stencilNormalVelocity[0], stencilWeights, dVeldNorm);
    PetscScalar dPdNorm;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryPressure, stencilSize, &stencilPressure[0], stencilWeights, dPdNorm);

    // Compute the temperature at the boundary
    PetscReal boundaryTemperature;
    inletBoundary->eos->GetComputeTemperatureFunction()(dim,
                                                         boundaryDensity,
                                                         boundaryValues[uOff[EULER] + fp::RHOE] / boundaryDensity,
                                                         boundaryValues + uOff[EULER] + fp::RHOU,
                                                         nullptr,
                                                         &boundaryTemperature,
                                                        inletBoundary->eos->GetComputeTemperatureContext()) >>
        checkError;

    // Compute the cp, cv from the eos
    PetscReal boundaryCp, boundaryCv;
    inletBoundary->eos->GetComputeSpecificHeatConstantPressureFunction()(
        boundaryTemperature, boundaryDensity, nullptr, &boundaryCp, inletBoundary->eos->GetComputeSpecificHeatConstantPressureContext()) >>
        checkError;
    inletBoundary->eos->GetComputeSpecificHeatConstantVolumeFunction()(
        boundaryTemperature, boundaryDensity, nullptr, &boundaryCv, inletBoundary->eos->GetComputeSpecificHeatConstantVolumeContext()) >>
        checkError;

    // Compute the enthalpy
    PetscReal boundarySensibleEnthalpy;
    inletBoundary->eos->GetComputeSensibleEnthalpyFunction()(boundaryTemperature, boundaryDensity, nullptr, &boundarySensibleEnthalpy, inletBoundary->eos->GetComputeSensibleEnthalpyContext()) >>
        checkError;

    // get_vel_and_c_prims(PGS, velwall[0], C, Cp, Cv, velnprm, Cprm);
    PetscReal velNormPrim, speedOfSoundPrim;
    GetVelAndCPrims(boundaryNormalVelocity, boundarySpeedOfSound, boundaryCp, boundaryCv, velNormPrim, speedOfSoundPrim);

    // get_eigenvalues
    std::vector<PetscReal> lambda(neq);
    GetEigenValues(dim, 0, 0, boundaryNormalVelocity, boundarySpeedOfSound, velNormPrim, speedOfSoundPrim, &lambda[0]);

    // Get scriptL
    std::vector<PetscReal> scriptL(neq);
    // Outgoing acoustic wave
    scriptL[1+dim] = lambda[1+dim]*(dPdNorm-boundaryDensity*dVeldNorm*(velNormPrim-boundaryNormalVelocity-speedOfSoundPrim));

    // Incoming acoustic wave
    scriptL[0] = scriptL[1 + dim];

    // Entropy wave
    scriptL[1] = 0.5e+0*(boundaryCp/boundaryCv-1.e+0)*(scriptL[1+dim]+scriptL[0])
                 -0.5*(boundaryCp/boundaryCv+1.e+0)*(scriptL[0]-scriptL[1+dim])
                       *(velNormPrim-boundaryNormalVelocity)/speedOfSoundPrim;

    // Tangential velocities
    for (int d = 1; d < dim; d++) {
        scriptL[1 + d] = 0.e+0;
    }
    //    for (int ns = 0; ns < nspeceq; ns++) {
    //        sL[2 + ndims + ns][n1][n0] = 0.e+0; // Species
    //    }
    //    for (int ne = 0; ne < nEVeq; ne++) {
    //        sL[2 + ndims + nspeceq + ne][n1][n0] = 0.e+0; // Extra variables
    //    }

    // Directly compute the source terms, note that this may be problem in the future with multiple source terms on the same boundary cell
    GetmdFdn(dim,
             neq,
             0,
             0,
             boundaryVelNormCord,
             boundaryDensity,
             boundaryTemperature,
             boundaryCp,
             boundaryCv,
             boundarySpeedOfSound,
             boundarySensibleEnthalpy,
             velNormPrim,
             speedOfSoundPrim,
             nullptr /* PetscReal* Yi*/,
             nullptr /* PetscReal* EV*/,
             &scriptL[0],
             transformationMatrix,
             source + sOff[EULER]);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::lodi::Inlet, "Enforces an inlet with specified velocity",
         ARG(ablate::eos::EOS, "eos", "The EOS describing the flow field at the wall"));

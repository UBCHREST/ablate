#include "openBoundary.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"

using fp = ablate::finiteVolume::processes::FlowProcess;

ablate::boundarySolver::lodi::OpenBoundary::OpenBoundary(std::shared_ptr<eos::EOS> eos, double reflectFactor, double referencePressure, double maxAcousticsLength)
    : LODIBoundary(std::move(eos)), reflectFactor((PetscReal)reflectFactor), referencePressure((PetscReal)referencePressure), maxAcousticsLength((PetscReal)maxAcousticsLength) {}

void ablate::boundarySolver::lodi::OpenBoundary::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    ablate::boundarySolver::lodi::LODIBoundary::Initialize(bSolver);

    bSolver.RegisterFunction(OpenBoundaryFunction, this, fieldNames, fieldNames, {});
}

PetscErrorCode ablate::boundarySolver::lodi::OpenBoundary::OpenBoundaryFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell,
                                                                                const PetscInt *uOff, const PetscScalar *boundaryValues, const PetscScalar **stencilValues, const PetscInt *aOff,
                                                                                const PetscScalar *auxValues, const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                                const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    auto boundary = (OpenBoundary *)ctx;
    auto decodeStateFunction = boundary->eos->GetDecodeStateFunction();
    auto decodeStateContext = boundary->eos->GetDecodeStateContext();

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
                                                           boundaryValues + uOff[boundary->eulerId],
                                                           nullptr,
                                                           fg->normal,
                                                           &boundaryDensity,
                                                           &boundaryNormalVelocity,
                                                           boundaryVel,
                                                           &boundaryInternalEnergy,
                                                           &boundarySpeedOfSound,
                                                           &boundaryMach,
                                                           &boundaryPressure);

    boundaryMach = PetscAbs(boundaryMach);

    // Map the boundary velocity into the normal coord system
    PetscReal boundaryVelNormCord[3];
    utilities::MathUtilities::Multiply(dim, transformationMatrix, boundaryVel, boundaryVelNormCord);

    // Compute each stencil point
    std::vector<PetscReal> stencilDensity(stencilSize);
    std::vector<std::vector<PetscReal>> stencilVel(stencilSize, std::vector<PetscReal>(dim));
    std::vector<std::vector<PetscReal>> stencilNormalCoordsVel(dim, std::vector<PetscReal>(stencilSize));  // NOTE this is [dim][stencil]
    std::vector<PetscReal> stencilInternalEnergy(stencilSize);
    std::vector<PetscReal> stencilNormalVelocity(stencilSize);
    std::vector<PetscReal> stencilSpeedOfSound(stencilSize);
    std::vector<PetscReal> stencilMach(stencilSize);
    std::vector<PetscReal> stencilPressure(stencilSize);

    for (PetscInt s = 0; s < stencilSize; s++) {
        finiteVolume::processes::FlowProcess::DecodeEulerState(decodeStateFunction,
                                                               decodeStateContext,
                                                               dim,
                                                               &stencilValues[s][uOff[boundary->eulerId]],
                                                               nullptr,
                                                               fg->normal,
                                                               &stencilDensity[s],
                                                               &stencilNormalVelocity[s],
                                                               &stencilVel[s][0],
                                                               &stencilInternalEnergy[s],
                                                               &stencilSpeedOfSound[s],
                                                               &stencilMach[s],
                                                               &stencilPressure[s]);

        // Map the stencil velocity to a normal velocity
        PetscReal normalCoordsVel[3];
        utilities::MathUtilities::Multiply(dim, transformationMatrix, &stencilVel[s][0], normalCoordsVel);

        for (PetscInt d = 0; d < dim; d++) {
            stencilNormalCoordsVel[d][s] = normalCoordsVel[d];
        }
    }

    // Interpolate the normal velocity gradient to the surface
    PetscScalar dVeldNorm[3];
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryVelNormCord[0], stencilSize, &stencilNormalCoordsVel[0][0], stencilWeights, dVeldNorm[0]);
    for (PetscInt d = 1; d < dim; d++) {
        BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryVelNormCord[d], stencilSize, &stencilNormalCoordsVel[d][0], stencilWeights, dVeldNorm[d]);
    }
    PetscScalar dRhodNorm;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryDensity, stencilSize, &stencilDensity[0], stencilWeights, dRhodNorm);
    PetscScalar dPdNorm;
    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryPressure, stencilSize, &stencilPressure[0], stencilWeights, dPdNorm);

    // Compute the temperature at the boundary
    PetscReal boundaryTemperature;
    boundary->eos->GetComputeTemperatureFunction()(dim,
                                                   boundaryDensity,
                                                   boundaryValues[uOff[boundary->eulerId] + fp::RHOE] / boundaryDensity,
                                                   boundaryValues + uOff[boundary->eulerId] + fp::RHOU,
                                                   nullptr,
                                                   &boundaryTemperature,
                                                   boundary->eos->GetComputeTemperatureContext()) >>
        checkError;

    // Compute the cp, cv from the eos
    PetscReal boundaryCp, boundaryCv;
    boundary->eos->GetComputeSpecificHeatConstantPressureFunction()(boundaryTemperature, boundaryDensity, nullptr, &boundaryCp, boundary->eos->GetComputeSpecificHeatConstantPressureContext()) >>
        checkError;
    boundary->eos->GetComputeSpecificHeatConstantVolumeFunction()(boundaryTemperature, boundaryDensity, nullptr, &boundaryCv, boundary->eos->GetComputeSpecificHeatConstantVolumeContext()) >>
        checkError;

    // Compute the enthalpy
    PetscReal boundarySensibleEnthalpy;
    boundary->eos->GetComputeSensibleEnthalpyFunction()(boundaryTemperature, boundaryDensity, nullptr, &boundarySensibleEnthalpy, boundary->eos->GetComputeSensibleEnthalpyContext()) >> checkError;

    // get_vel_and_c_prims(PGS, velwall[0], C, Cp, Cv, velnprm, Cprm);
    PetscReal velNormPrim, speedOfSoundPrim;
    GetVelAndCPrims(boundaryNormalVelocity, boundarySpeedOfSound, boundaryCp, boundaryCv, velNormPrim, speedOfSoundPrim);

    // get_eigenvalues
    std::vector<PetscReal> lambda(boundary->nEqs);
    boundary->GetEigenValues(boundaryNormalVelocity, boundarySpeedOfSound, velNormPrim, speedOfSoundPrim, &lambda[0]);

    // compute the relaxation timescale
    // L2 = (p - pref)/tau = (p - pref)*kFac*a
    // kFac = rf/L
    PetscReal kFac = boundary->reflectFactor / boundary->maxAcousticsLength;

    // Compute scriptL
    std::vector<PetscReal> scriptL(boundary->nEqs);
    {
        if (boundaryMach < 1.0) {
            // Subsonic
            // Outgoing acoustic wave
            scriptL[1 + dim] = lambda[1 + dim] * (dPdNorm - boundaryDensity * dVeldNorm[0] * (velNormPrim - boundaryNormalVelocity - speedOfSoundPrim));
            // Incoming acoustic wave
            scriptL[0] = kFac * speedOfSoundPrim * (boundaryPressure - boundary->referencePressure);

            if (boundaryNormalVelocity >= 0.0) {
                // If going out of the domain
                scriptL[1] = lambda[1] * (PetscSqr(boundarySpeedOfSound) * dRhodNorm - dPdNorm + 2.e+0 * boundaryDensity * (velNormPrim - boundaryNormalVelocity) * dVeldNorm[0]);
                for (int d = 1; d < dim; d++) {
                    scriptL[1 + d] = lambda[1 + d] * dVeldNorm[d];  // Tangential velocities
                };
                //                                for (int ns = 0; ns < nspeceq; ns++) {
                //                                    scriptL[2+dim+ns] = lambda[2+ndims+ns]*dYidn[ns][n1][n0];// Species
                //                                }
                //                                for (int ne = 0; ne < nEVeq; ne++) {
                //                                    scriptL[2+dim+nspeceq+ne] = lambda[2+ndims+nspeceq+ne]*dEVdn[ne][n1][n0];// Scalars
                //                                }
            } else {
                // Coming into the domain (assume dP/dt = 0)
                scriptL[1] = 0.;  // Entropy wave
                for (int d = 1; d < dim; d++) {
                    scriptL[1 + d] = 0.e+0;  // Tangential velocities
                };
                //                for (int ns = 0; ns < nspeceq; ns++) {
                //                    scriptL[2+ndims+ns][n1][n0] = 0.;// Species
                //                }
                //                for (int ne = 0; ne < nEVeq; ne++) {
                //                    scriptL[2+ndims+nspeceq+ne][n1][n0] = 0.; // Scalars
                //                }
            }
        } else {
            // Supersonic
            // Going out of the domain
            if (boundaryNormalVelocity >= 0.e+0) {
                scriptL[0] = lambda[0] * (dPdNorm - boundaryDensity * dVeldNorm[0] * (velNormPrim - boundaryNormalVelocity + speedOfSoundPrim));  // Outgoing acoustic wave
                double tmp2 = boundarySpeedOfSound * boundarySpeedOfSound;
                scriptL[1] = lambda[1] * (tmp2 * dRhodNorm - dPdNorm + 2.0 * boundaryDensity * (velNormPrim - boundaryNormalVelocity) * dVeldNorm[0]);
                for (int d = 1; d < dim; d++) {
                    scriptL[1 + d] = lambda[1 + d] * dVeldNorm[d];
                }
                scriptL[1 + dim] = lambda[1 + dim] * (dPdNorm - boundaryDensity * dVeldNorm[0] * (velNormPrim - boundaryNormalVelocity - speedOfSoundPrim));
                //                for (int ns = 0; ns < nspeceq; ns++) {
                //                    sL[2+ndims+ns][n1][n0] = lam[2+ndims+ns]*dYidn[ns][n1][n0];// Species
                //                }
                //                for (int ne = 0; ne < nEVeq; ne++) {
                //                    sL[2+ndims+nspeceq+ne][n1][n0] = lam[2+ndims+nspeceq+ne]*dEVdn[ne][n1][n0];// Scalars
                //                }
            }
            // Coming into the domain
            else {
                scriptL[1] = 0.;  // Entropy wave
                for (int d = 1; d < dim; d++) {
                    scriptL[1 + d] = 0.e+0;  // Tangential velocities
                };
                //                for (int ns = 0; ns < nspeceq; ns++) {
                //                    sL[2+ndims+ns][n1][n0] = 0.;  					// Species
                //                }
                //                for (int ne = 0; ne < nEVeq; ne++) {
                //                    sL[2+ndims+nspeceq+ne][n1][n0] = 0.;  			// Scalars
                //                }
            }
        }
    }

    // Directly compute the source terms, note that this may be problem in the future with multiple source terms on the same boundary cell
    boundary->GetmdFdn(sOff,
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
                       source);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::lodi::OpenBoundary, "Treats boundary as open.",
         ARG(ablate::eos::EOS, "eos", "The EOS describing the flow field at the boundary"), ARG(double, "reflectFactor", "boundary reflection factor"),
         ARG(double, "referencePressure", "reference pressure"), ARG(double, "maxAcousticsLength", "maximum length in the domain for acoustics to propagate "));

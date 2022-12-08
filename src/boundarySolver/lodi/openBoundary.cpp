#include "openBoundary.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"

using fp = ablate::finiteVolume::CompressibleFlowFields;

ablate::boundarySolver::lodi::OpenBoundary::OpenBoundary(std::shared_ptr<eos::EOS> eos, double reflectFactor, double referencePressure, double maxAcousticsLength,
                                                         std::shared_ptr<finiteVolume::processes::PressureGradientScaling> pressureGradientScaling)
    : LODIBoundary(std::move(eos), std::move(pressureGradientScaling)),
      reflectFactor((PetscReal)reflectFactor),
      referencePressure((PetscReal)referencePressure),
      maxAcousticsLength((PetscReal)maxAcousticsLength) {}

void ablate::boundarySolver::lodi::OpenBoundary::Setup(ablate::boundarySolver::BoundarySolver &bSolver) {
    ablate::boundarySolver::lodi::LODIBoundary::Setup(bSolver);

    bSolver.RegisterFunction(OpenBoundaryFunction, this, fieldNames, fieldNames, {});
}

PetscErrorCode ablate::boundarySolver::lodi::OpenBoundary::OpenBoundaryFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell,
                                                                                const PetscInt *uOff, const PetscScalar *boundaryValues, const PetscScalar **stencilValues, const PetscInt *aOff,
                                                                                const PetscScalar *auxValues, const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                                const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    auto boundary = (OpenBoundary *)ctx;

    // Compute the transformation matrix
    PetscReal transformationMatrix[3][3];
    utilities::MathUtilities::ComputeTransformationMatrix(dim, fg->normal, transformationMatrix);

    // Compute the pressure/values on the boundary
    PetscReal boundaryDensity;
    PetscReal boundaryTemperature;
    PetscReal boundaryVel[3];
    PetscReal boundaryNormalVelocity = 0.0;
    PetscReal boundarySpeedOfSound;
    PetscReal boundaryMach;
    PetscReal boundaryPressure;

    // Get the densityYi pointer if available
    const PetscScalar *boundaryDensityYi = boundary->nSpecEqs > 0 ? boundaryValues + uOff[boundary->speciesId] : nullptr;

    // Get the velocity and pressure on the surface
    {
        boundaryDensity = boundaryValues[uOff[boundary->eulerId] + finiteVolume::CompressibleFlowFields::RHO];
        for (PetscInt d = 0; d < dim; d++) {
            boundaryVel[d] = boundaryValues[uOff[boundary->eulerId] + finiteVolume::CompressibleFlowFields::RHOU + d] / boundaryDensity;
            boundaryNormalVelocity += boundaryVel[d] * fg->normal[d];
        }
        PetscErrorCode ierr = boundary->computeTemperature.function(boundaryValues, &boundaryTemperature, boundary->computeTemperature.context.get());
        CHKERRQ(ierr);
        ierr = boundary->computeSpeedOfSound.function(boundaryValues, boundaryTemperature, &boundarySpeedOfSound, boundary->computeSpeedOfSound.context.get());
        CHKERRQ(ierr);
        ierr = boundary->computePressureFromTemperature.function(boundaryValues, boundaryTemperature, &boundaryPressure, boundary->computePressureFromTemperature.context.get());
        CHKERRQ(ierr);
        boundaryMach = PetscAbs(boundaryNormalVelocity / boundarySpeedOfSound);
    }

    // Map the boundary velocity into the normal coord system
    PetscReal boundaryVelNormCord[3];
    utilities::MathUtilities::Multiply(dim, transformationMatrix, boundaryVel, boundaryVelNormCord);

    // Compute each stencil point
    std::vector<PetscReal> stencilDensity(stencilSize);
    std::vector<std::vector<PetscReal>> stencilVel(stencilSize, std::vector<PetscReal>(dim));
    std::vector<std::vector<PetscReal>> stencilNormalCoordsVel(dim, std::vector<PetscReal>(stencilSize));  // NOTE this is [dim][stencil]
    std::vector<PetscReal> stencilNormalVelocity(stencilSize, 0.0);
    std::vector<PetscReal> stencilPressure(stencilSize);
    std::vector<std::vector<PetscReal>> stencilYi(boundary->nSpecEqs, std::vector<PetscReal>(stencilSize));  // NOTE this is [sp][stencil]
    std::vector<std::vector<PetscReal>> stencilEv(boundary->nEvEqs, std::vector<PetscReal>(stencilSize));    // NOTE this is [sp][stencil]

    for (PetscInt s = 0; s < stencilSize; s++) {
        stencilDensity[s] = stencilValues[s][uOff[boundary->eulerId] + finiteVolume::CompressibleFlowFields::RHO];
        for (PetscInt d = 0; d < dim; d++) {
            stencilVel[s][d] = stencilValues[s][uOff[boundary->eulerId] + finiteVolume::CompressibleFlowFields::RHOU + d] / stencilDensity[s];
            stencilNormalVelocity[s] += stencilVel[s][d] * fg->normal[d];
        }
        PetscErrorCode ierr = boundary->computePressure.function(stencilValues[s], &stencilPressure[s], boundary->computePressure.context.get());
        CHKERRQ(ierr);

        // Map the stencil velocity to a normal velocity
        PetscReal normalCoordsVel[3];
        utilities::MathUtilities::Multiply(dim, transformationMatrix, &stencilVel[s][0], normalCoordsVel);

        for (PetscInt d = 0; d < dim; d++) {
            stencilNormalCoordsVel[d][s] = normalCoordsVel[d];
        }

        // Compute each of the species and ev
        for (PetscInt sp = 0; sp < boundary->nSpecEqs; sp++) {
            stencilYi[sp][s] = stencilValues[s][uOff[boundary->speciesId] + sp] / stencilDensity[s];
        }
        int ne = 0;
        for (std::size_t ev = 0; ev < boundary->evIds.size(); ++ev) {
            for (PetscInt ec = 0; ec < boundary->nEvComps[ev]; ++ec) {
                stencilEv[ne++][s] = stencilValues[s][uOff[boundary->evIds[ev]] + ec] / stencilDensity[s];
            }
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

    // compute boundary ev, yi
    std::vector<PetscReal> boundaryYi(boundary->nSpecEqs);
    for (PetscInt i = 0; i < boundary->nSpecEqs; i++) {
        boundaryYi[i] = boundaryDensityYi[i] / boundaryDensity;
    }
    std::vector<PetscReal> boundaryEv(boundary->nEvEqs);
    int i = 0;
    for (std::size_t ev = 0; ev < boundary->evIds.size(); ++ev) {
        const PetscReal *rhoEV = boundaryValues + uOff[boundary->evIds[ev]];
        for (PetscInt ec = 0; ec < boundary->nEvComps[ev]; ++ec) {
            boundaryEv[i] = rhoEV[ec] / boundaryDensity;
            i++;
        }
    }

    // Compute the cp, cv from the eos
    PetscReal boundaryCp, boundaryCv;
    boundary->computeSpecificHeatConstantPressure.function(boundaryValues, boundaryTemperature, &boundaryCp, boundary->computeSpecificHeatConstantPressure.context.get());
    boundary->computeSpecificHeatConstantVolume.function(boundaryValues, boundaryTemperature, &boundaryCv, boundary->computeSpecificHeatConstantVolume.context.get());

    // Compute the enthalpy
    PetscReal boundarySensibleEnthalpy;
    boundary->computeSensibleEnthalpyFunction.function(boundaryValues, boundaryTemperature, &boundarySensibleEnthalpy, boundary->computeSensibleEnthalpyFunction.context.get());

    // get_vel_and_c_prims(PGS, velwall[0], C, Cp, Cv, velnprm, Cprm);
    PetscReal velNormPrim, speedOfSoundPrim;
    boundary->GetVelAndCPrims(boundaryNormalVelocity, boundarySpeedOfSound, boundaryCp, boundaryCv, velNormPrim, speedOfSoundPrim);

    // get_eigenvalues
    std::vector<PetscReal> lambda(boundary->nEqs);
    boundary->GetEigenValues(boundaryNormalVelocity, boundarySpeedOfSound, velNormPrim, speedOfSoundPrim, &lambda[0]);

    // compute the relaxation timescale
    // L2 = (p - pref)/tau = (p - pref)*kFac*a
    // kFac = rf/L
    PetscReal kFac = boundary->reflectFactor / boundary->maxAcousticsLength;

    // Compute alpha2
    PetscReal alpha2 = 1.0;
    if (boundary->pressureGradientScaling) {
        alpha2 = PetscSqr(boundary->pressureGradientScaling->GetAlpha());
    }

    // Compute scriptL
    std::vector<PetscReal> scriptL(boundary->nEqs);
    {
        if (boundaryMach < 1.0) {
            // Subsonic
            // Outgoing acoustic wave
            scriptL[1 + dim] = lambda[1 + dim] * (dPdNorm - boundaryDensity * alpha2 * dVeldNorm[0] * (velNormPrim - boundaryNormalVelocity - speedOfSoundPrim));
            // Incoming acoustic wave
            scriptL[0] = kFac * speedOfSoundPrim * (boundaryPressure - boundary->referencePressure);

            if (boundaryNormalVelocity >= 0.0) {
                // If going out of the domain
                scriptL[1] = lambda[1] * (PetscSqr(boundarySpeedOfSound) * dRhodNorm - dPdNorm + 2.e+0 * boundaryDensity * alpha2 * (velNormPrim - boundaryNormalVelocity) * dVeldNorm[0]);
                for (int d = 1; d < dim; d++) {
                    scriptL[1 + d] = lambda[1 + d] * dVeldNorm[d];  // Tangential velocities
                };
                for (int ns = 0; ns < boundary->nSpecEqs; ns++) {
                    PetscScalar dYidn;
                    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryYi[ns], stencilSize, stencilYi[ns].data(), stencilWeights, dYidn);
                    scriptL[2 + dim + ns] = lambda[2 + dim + ns] * dYidn;  // Species
                }
                for (int ne = 0; ne < boundary->nEvEqs; ne++) {
                    PetscScalar dEvdn;
                    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryEv[ne], stencilSize, stencilEv[ne].data(), stencilWeights, dEvdn);

                    scriptL[2 + dim + boundary->nSpecEqs + ne] = lambda[2 + dim + boundary->nSpecEqs + ne] * dEvdn;  // Scalars
                }
            } else {
                // Coming into the domain (assume dP/dt = 0)
                scriptL[1] = 0.;  // Entropy wave
                for (int d = 1; d < dim; d++) {
                    scriptL[1 + d] = 0.e+0;  // Tangential velocities
                };
                for (int ns = 0; ns < boundary->nSpecEqs; ns++) {
                    scriptL[2 + dim + ns] = 0.;  // Species
                }
                for (int ne = 0; ne < boundary->nEvEqs; ne++) {
                    scriptL[2 + dim + boundary->nSpecEqs + ne] = 0.;  // Scalars
                }
            }
        } else {
            // Supersonic
            // Going out of the domain
            if (boundaryNormalVelocity >= 0.e+0) {
                scriptL[0] = lambda[0] * (dPdNorm - boundaryDensity * alpha2 * dVeldNorm[0] * (velNormPrim - boundaryNormalVelocity + speedOfSoundPrim));  // Outgoing acoustic wave
                double tmp2 = boundarySpeedOfSound * boundarySpeedOfSound;
                scriptL[1] = lambda[1] * (tmp2 * dRhodNorm - dPdNorm + 2.0 * boundaryDensity * alpha2 * (velNormPrim - boundaryNormalVelocity) * dVeldNorm[0]);
                for (int d = 1; d < dim; d++) {
                    scriptL[1 + d] = lambda[1 + d] * dVeldNorm[d];
                }
                scriptL[1 + dim] = lambda[1 + dim] * (dPdNorm - boundaryDensity * alpha2 * dVeldNorm[0] * (velNormPrim - boundaryNormalVelocity - speedOfSoundPrim));
                for (int ns = 0; ns < boundary->nSpecEqs; ns++) {
                    PetscScalar dYidn;
                    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryYi[ns], stencilSize, stencilYi[ns].data(), stencilWeights, dYidn);

                    scriptL[2 + dim + ns] = lambda[2 + dim + ns] * dYidn;  // Species
                }
                for (int ne = 0; ne < boundary->nEvEqs; ne++) {
                    PetscScalar dEvdn;
                    BoundarySolver::ComputeGradientAlongNormal(dim, fg, boundaryEv[ne], stencilSize, stencilEv[ne].data(), stencilWeights, dEvdn);

                    scriptL[2 + dim + boundary->nSpecEqs + ne] = lambda[2 + dim + boundary->nSpecEqs + ne] * dEvdn;  // Scalars
                }
            }
            // Coming into the domain
            else {
                scriptL[1] = 0.;  // Entropy wave
                for (int d = 1; d < dim; d++) {
                    scriptL[1 + d] = 0.e+0;  // Tangential velocities
                };
                for (int ns = 0; ns < boundary->nSpecEqs; ns++) {
                    scriptL[2 + dim + ns] = 0.;  // Species
                }
                for (int ne = 0; ne < boundary->nEvEqs; ne++) {
                    scriptL[2 + dim + boundary->nSpecEqs + ne] = 0.;  // Scalars
                }
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
                       boundaryValues,
                       uOff,
                       scriptL.data(),
                       transformationMatrix,
                       source);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::lodi::OpenBoundary, "Treats boundary as open.",
         ARG(ablate::eos::EOS, "eos", "The EOS describing the flow field at the boundary"), ARG(double, "reflectFactor", "boundary reflection factor"),
         ARG(double, "referencePressure", "reference pressure"), ARG(double, "maxAcousticsLength", "maximum length in the domain for acoustics to propagate "),
         OPT(ablate::finiteVolume::processes::PressureGradientScaling, "pgs", "Pressure gradient scaling is used to scale the acoustic propagation speed and increase time step for low speed flows"));

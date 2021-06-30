#include "compressibleFlow.h"
#include "fvSupport.h"

static inline void NormVector(PetscInt dim, const PetscReal* in, PetscReal* out){
    PetscReal mag = 0.0;
    for (PetscInt d=0; d< dim; d++) {
        mag += in[d]*in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d=0; d< dim; d++) {
        out[d] = in[d]/mag;
    }
}

static inline PetscReal MagVector(PetscInt dim, const PetscReal* in){
    PetscReal mag = 0.0;
    for (PetscInt d=0; d< dim; d++) {
        mag += in[d]*in[d];
    }
    return PetscSqrtReal(mag);
}


/**
 * Function to get the density, velocity, and energy from the conserved variables
 * @return
 */
static void DecodeEulerState(FlowData_CompressibleFlow flowData, PetscInt dim, const PetscReal* conservedValues, const PetscReal* densityYi, const PetscReal *normal, PetscReal* density,
                                  PetscReal* normalVelocity, PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* M, PetscReal* p){
    // decode
    *density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE]/(*density);

    // Get the velocity in this direction
    (*normalVelocity) = 0.0;
    for (PetscInt d =0; d < dim; d++){
        velocity[d] = conservedValues[RHOU + d]/(*density);
        (*normalVelocity) += velocity[d]*normal[d];
    }

    // decode the state in the eos
    flowData->decodeStateFunction(dim, *density, totalEnergy, velocity, densityYi, internalEnergy, a, p, flowData->decodeStateFunctionContext);
    *M = (*normalVelocity)/(*a);
}

PetscErrorCode CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVelL, const PetscReal * gradVelR, PetscReal* tau){
    PetscFunctionBeginUser;
    // pre compute the div of the velocity field
    PetscReal divVel = 0.0;
    for (PetscInt c =0; c < dim; ++c){
        divVel += 0.5*(gradVelL[c*dim + c] + gradVelR[c*dim + c]);
    }

    // March over each velocity component, u, v, w
    for (PetscInt c =0; c < dim; ++c){
        // March over each physical coordinate coordinate
        for (PetscInt d =0; d < dim; ++d) {
            if (d == c) {
                // for the xx, yy, zz, components
                tau[c*dim + d] = 2.0 * mu * (0.5 * (gradVelL[c * dim + d] + gradVelR[c * dim + d]) - divVel / 3.0);
            } else {
                // for xy, xz, etc
                tau[c*dim + d]  = mu *( 0.5 * (gradVelL[c * dim + d] + gradVelR[c * dim + d]) + 0.5 * (gradVelL[d * dim + c] + gradVelR[d * dim + c]));
            }
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode CompressibleFlowComputeEulerFlux ( PetscInt dim, const PetscFVFaceGeom* fg,
                                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar uL[], const PetscScalar uR[], const PetscScalar gradL[], const PetscScalar gradR[],
                                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                                  PetscScalar* flux, void* ctx){
    FlowData_CompressibleFlow flowParameters = (FlowData_CompressibleFlow)ctx;
    PetscFunctionBeginUser;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal ML;
    PetscReal pL;
    const PetscReal *densityYiL = flowParameters->numberSpecies > 0? uL + uOff[1] : NULL;
    DecodeEulerState(flowParameters, dim, uL + uOff[0],densityYiL, norm, &densityL, &normalVelocityL, velocityL, &internalEnergyL, &aL, &ML, &pL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal MR;
    PetscReal pR;
    const PetscReal *densityYiR = flowParameters->numberSpecies > 0? uR + uOff[1] : NULL;
    DecodeEulerState(flowParameters, dim, uR + uOff[0],densityYiR, norm, &densityR, &normalVelocityR, velocityR, &internalEnergyR, &aR, &MR, &pR);

    PetscReal sPm;
    PetscReal sPp;
    PetscReal sMm;
    PetscReal sMp;

    flowParameters->fluxDifferencer(MR, &sPm, &sMm, ML, &sPp, &sMp);

    flux[RHO] = (sMm* densityR * aR + sMp* densityL * aL) * areaMag;

    PetscReal velMagR = MagVector(dim, velocityR);
    PetscReal HR = internalEnergyR + velMagR*velMagR/2.0 + pR/densityR;
    PetscReal velMagL = MagVector(dim, velocityL);
    PetscReal HL = internalEnergyL + velMagL*velMagL/2.0 + pL/densityL;

    flux[RHOE] = (sMm * densityR * aR * HR + sMp * densityL * aL * HL) * areaMag;

    for (PetscInt n =0; n < dim; n++) {
        flux[RHOU + n] = (sMm * densityR * aR * velocityR[n] + sMp * densityL * aL * velocityL[n]) * areaMag + (pR*sPm + pL*sPp) * fg->normal[n];
    }

    PetscFunctionReturn(0);
}

PetscErrorCode CompressibleFlowEulerDiffusion(PetscInt dim, const PetscFVFaceGeom* fg,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar* fieldL, const PetscScalar* fieldR, const PetscScalar gradL[], const PetscScalar gradR[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar* auxL, const PetscScalar* auxR, const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                              PetscScalar* flux, void* ctx){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    FlowData_CompressibleFlow flowParameters = (FlowData_CompressibleFlow)ctx;

    // Compute the stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    ierr = CompressibleFlowComputeStressTensor(dim, flowParameters->mu, gradAuxL + aOff_x[VEL], gradAuxR + aOff_x[VEL], tau);CHKERRQ(ierr);

    // for each velocity component
    for (PetscInt c = 0; c < dim; ++c) {
        PetscReal viscousFlux = 0.0;

        // March over each direction
        for (PetscInt d = 0; d < dim; ++d) {
            viscousFlux += -fg->normal[d] * tau[c * dim + d];  // This is tau[c][d]
        }

        // add in the contribution
        flux[RHOU + c] = viscousFlux;
    }

    // energy equation
    flux[RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal heatFlux = 0.0;
        // add in the contributions for this viscous terms
        for (PetscInt c = 0; c < dim; ++c) {
            heatFlux += 0.5 * (auxL[aOff[VEL] + c] + auxR[aOff[VEL] + c]) * tau[d * dim + c];
        }

        // heat conduction (-k dT/dx - k dT/dy - k dT/dz) . n A
        heatFlux += +flowParameters->k * 0.5 * (gradAuxL[aOff_x[T] + d] + gradAuxR[aOff_x[T] + d]);

        // Multiply by the area normal
        heatFlux *= -fg->normal[d];

        flux[RHOE] += heatFlux;
    }
    PetscFunctionReturn(0);
}


PetscErrorCode CompressibleFlowSpeciesAdvectionFlux ( PetscInt dim, const PetscFVFaceGeom* fg,
                                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar uL[], const PetscScalar uR[], const PetscScalar gradL[], const PetscScalar gradR[],
                                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                                  PetscScalar* flux, void* ctx){
    FlowData_CompressibleFlow flowParameters = (FlowData_CompressibleFlow)ctx;
    PetscFunctionBeginUser;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    const int EULER_FIELD = 0;
    const int YI_FIELD = 1;

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal ML;
    PetscReal pL;
    DecodeEulerState(flowParameters, dim, uL + uOff[EULER_FIELD], uL + uOff[YI_FIELD], norm, &densityL, &normalVelocityL, velocityL, &internalEnergyL, &aL, &ML, &pL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal MR;
    PetscReal pR;
    DecodeEulerState(flowParameters, dim, uR + uOff[EULER_FIELD],uR + uOff[YI_FIELD], norm, &densityR, &normalVelocityR, velocityR, &internalEnergyR, &aR, &MR, &pR);

    PetscReal sPm;
    PetscReal sPp;
    PetscReal sMm;
    PetscReal sMp;

    flowParameters->fluxDifferencer(MR, &sPm, &sMm, ML, &sPp, &sMp);

    // march over each gas species
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; sp++){
        // Note: there is no density in the flux because uR and UL are density*yi
        flux[sp] = (sMm  * aR * uR[uOff[YI_FIELD]+ sp] + sMp  * aL * uL[uOff[YI_FIELD] + sp]) * areaMag;
    }

    PetscFunctionReturn(0);
}
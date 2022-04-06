#include "twoPhaseEulerAdvection.hpp"
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "flowProcess.hpp"

static inline void NormVector(PetscInt dim, const PetscReal *in, PetscReal *out) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d = 0; d < dim; d++) {
        out[d] = in[d] / mag;
    }
}

static inline PetscReal MagVector(PetscInt dim, const PetscReal *in) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    return PetscSqrtReal(mag);
}

ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidGas,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid)
    : eosGas(eosGas),
      eosLiquid(eosLiquid),
      fluxCalculatorGasGas(fluxCalculatorGasGas),
      fluxCalculatorGasLiquid(fluxCalculatorGasLiquid),
      fluxCalculatorLiquidGas(fluxCalculatorLiquidGas),
      fluxCalculatorLiquidLiquid(fluxCalculatorLiquidLiquid) {}

void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    // Create the decoder based upon the eoses
    decoder = CreateTwoPhaseDecoder(flow.GetSubDomain().GetDimensions(), eosGas, eosLiquid);

    // Currently, no option for species advection
    flow.RegisterRHSFunction(CompressibleFlowComputeEulerFlux, this, "euler", {"densityVF", "euler"}, {});
    flow.RegisterRHSFunction(CompressibleFlowComputeVFFlux, this, "densityVF", {"densityVF", "euler"}, {});

    // check to see if auxFieldUpdates needed to be added
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::VELOCITY_FIELD)) {
        flow.RegisterAuxFieldUpdate(UpdateAuxVelocityField2Gas, nullptr, std::vector<std::string>{CompressibleFlowFields::VELOCITY_FIELD}, {CompressibleFlowFields::EULER_FIELD});
    }
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::TEMPERATURE_FIELD)) {
        // add in aux update variables
        flow.RegisterAuxFieldUpdate(UpdateAuxTemperatureField2Gas, this, std::vector<std::string>{CompressibleFlowFields::TEMPERATURE_FIELD}, {"densityVF", "euler"});
    }
    if (flow.GetSubDomain().ContainsField("pressure")) {
        // add in aux update variables
        flow.RegisterAuxFieldUpdate(UpdateAuxPressureField2Gas, this, std::vector<std::string>{"pressure"}, {"densityVF", "euler"});
    }
    if (flow.GetSubDomain().ContainsField("volumeFraction")) {
        flow.RegisterAuxFieldUpdate(UpdateAuxVolumeFractionField2Gas, this, std::vector<std::string>{"volumeFraction"}, {"densityVF", "euler"});
    }
}
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x,
                                                                                                         const PetscScalar *fieldL, const PetscScalar *fieldR, const PetscScalar *gradL,
                                                                                                         const PetscScalar *gradR, const PetscInt *aOff, const PetscInt *aOff_x,
                                                                                                         const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                                         const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;
    // Compute the norm of cell face
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    // Decode left and right states
    PetscReal densityL;
    PetscReal densityG_L;
    PetscReal densityL_L;
    PetscReal normalVelocityL;  // uniform velocity in cell
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal internalEnergyG_L;
    PetscReal internalEnergyL_L;
    PetscReal aG_L;
    PetscReal aL_L;
    PetscReal MG_L;
    PetscReal ML_L;
    PetscReal pL;  // pressure equilibrium
    PetscReal tL;
    PetscReal alphaL;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(dim,
                                                              uOff,
                                                              fieldL,
                                                              norm,
                                                              &densityL,
                                                              &densityG_L,
                                                              &densityL_L,
                                                              &normalVelocityL,
                                                              velocityL,
                                                              &internalEnergyL,
                                                              &internalEnergyG_L,
                                                              &internalEnergyL_L,
                                                              &aG_L,
                                                              &aL_L,
                                                              &MG_L,
                                                              &ML_L,
                                                              &pL,
                                                              &tL,
                                                              &alphaL);

    PetscReal densityR;
    PetscReal densityG_R;
    PetscReal densityL_R;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal internalEnergyG_R;
    PetscReal internalEnergyL_R;
    PetscReal aG_R;
    PetscReal aL_R;
    PetscReal MG_R;
    PetscReal ML_R;
    PetscReal pR;
    PetscReal tR;
    PetscReal alphaR;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(dim,
                                                              uOff,
                                                              fieldR,
                                                              norm,
                                                              &densityR,
                                                              &densityG_R,
                                                              &densityL_R,
                                                              &normalVelocityR,
                                                              velocityR,
                                                              &internalEnergyR,
                                                              &internalEnergyG_R,
                                                              &internalEnergyL_R,
                                                              &aG_R,
                                                              &aL_R,
                                                              &MG_R,
                                                              &ML_R,
                                                              &pR,
                                                              &tR,
                                                              &alphaR);

    // get the face values
    PetscReal massFluxGG;
    PetscReal massFluxGL;
    PetscReal massFluxLL;
    PetscReal p12;  // pressure equilibrium ( might need to make sure they match)

    // call flux calculator 3 times, gas-gas, gas-liquid, liquid-liquid regions
    fluxCalculator::Direction directionG = twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFluxGG, &p12);
    // should be same direction, if not, big problem
    twoPhaseEulerAdvection->fluxCalculatorLiquidLiquid->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorLiquidLiquid->GetFluxCalculatorContext(), normalVelocityL, aL_L, densityL_L, pL, normalVelocityR, aL_R, densityL_R, pR, &massFluxLL, &p12);

    if (alphaL > alphaR) {
        // gas on left, liquid on right
        twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorFunction()(
            twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aL_R, densityL_R, pR, &massFluxGL, &p12);
    } else if (alphaL < alphaR) {
        // liquid on left, gas on right
        twoPhaseEulerAdvection->fluxCalculatorLiquidGas->GetFluxCalculatorFunction()(
            twoPhaseEulerAdvection->fluxCalculatorLiquidGas->GetFluxCalculatorContext(), normalVelocityL, aL_L, densityL_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFluxGL, &p12);
    } else {
        // no discontinuous region
        massFluxGL = 0.0;
    }

    // Calculate total flux
    PetscReal alphaMin = PetscMin(alphaR, alphaL);
    PetscReal alphaDif = PetscAbs(alphaL - alphaR);
    if (directionG == fluxCalculator::LEFT) {  // direction of GG,LL,LG should match since uniform velocity???
                                               //        flux[RHO] = massFlux * areaMag;
        flux[CompressibleFlowFields::RHO] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * areaMag * (1 - alphaMin - alphaDif));
        PetscReal velMagL = MagVector(dim, velocityL);
        PetscReal HG_L = internalEnergyG_L + velMagL * velMagL / 2.0 + pL / densityG_L;
        PetscReal HL_L = internalEnergyL_L + velMagL * velMagL / 2.0 + pL / densityL_L;
        PetscReal HGL_L;
        if (alphaL > alphaR) {
            // gas on left
            HGL_L = internalEnergyG_L + velMagL * velMagL / 2.0 + pL / densityG_L;
        } else if (alphaL < alphaR) {
            // liquid on left
            HGL_L = internalEnergyL_L + velMagL * velMagL / 2.0 + pL / densityL_L;
        } else {
            // no discontinuous region
            HGL_L = 0.0;
        }
        //        flux[RHOE] = HL * massFlux * areaMag;
        flux[CompressibleFlowFields::RHOE] = (HG_L * massFluxGG * areaMag * alphaMin) + (HGL_L * massFluxGL * areaMag * alphaDif) + (HL_L * massFluxLL * areaMag * (1 - alphaMin - alphaDif));
        for (PetscInt n = 0; n < dim; n++) {
            //            flux[RHOU + n] = velocityL[n] * massFlux * areaMag + p12 * fg->normal[n];
            flux[CompressibleFlowFields::RHOU + n] = velocityL[n] * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1 - alphaMin - alphaDif)) + p12 * fg->normal[n];
        }
    } else if (directionG == fluxCalculator::RIGHT) {
        //        flux[RHO] = massFlux * areaMag;
        flux[CompressibleFlowFields::RHO] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * areaMag * (1 - alphaMin - alphaDif));
        PetscReal velMagR = MagVector(dim, velocityR);
        PetscReal HG_R = internalEnergyG_R + velMagR * velMagR / 2.0 + pR / densityG_R;
        PetscReal HL_R = internalEnergyL_R + velMagR * velMagR / 2.0 + pR / densityL_R;
        PetscReal HGL_R;
        if (alphaL > alphaR) {
            // liquid on right
            HGL_R = internalEnergyL_R + velMagR * velMagR / 2.0 + pR / densityL_R;
        } else if (alphaL < alphaR) {
            // gas on right
            HGL_R = internalEnergyG_R + velMagR * velMagR / 2.0 + pR / densityG_R;
        } else {
            // no discontinuous region
            HGL_R = 0.0;
        }
        //        flux[RHOE] = HR * massFlux * areaMag;
        flux[CompressibleFlowFields::RHOE] = (HG_R * massFluxGG * areaMag * alphaMin) + (HGL_R * massFluxGL * areaMag * alphaDif) + (HL_R * massFluxLL * areaMag * (1 - alphaMin - alphaDif));
        for (PetscInt n = 0; n < dim; n++) {
            //            flux[RHOU + n] = velocityR[n] * massFlux * areaMag + p12 * fg->normal[n];
            flux[CompressibleFlowFields::RHOU + n] = velocityR[n] * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1 - alphaMin - alphaDif)) + p12 * fg->normal[n];
        }
    } else {
        //        flux[RHO] = massFlux * areaMag;
        flux[CompressibleFlowFields::RHO] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * areaMag * (1 - alphaMin - alphaDif));

        PetscReal velMagL = MagVector(dim, velocityL);
        //        PetscReal HL = internalEnergyL + velMagL * velMagL / 2.0 + pL / densityL;
        PetscReal HG_L = internalEnergyG_L + velMagL * velMagL / 2.0 + pL / densityG_L;
        PetscReal HL_L = internalEnergyL_L + velMagL * velMagL / 2.0 + pL / densityL_L;
        PetscReal HGL_L;
        if (alphaL > alphaR) {
            // gas on left
            HGL_L = internalEnergyG_L + velMagL * velMagL / 2.0 + pL / densityG_L;
        } else if (alphaL < alphaR) {
            // liquid on left
            HGL_L = internalEnergyL_L + velMagL * velMagL / 2.0 + pL / densityL_L;
        } else {
            // no discontinuous region
            HGL_L = 0.0;
        }

        PetscReal velMagR = MagVector(dim, velocityR);
        //        PetscReal HR = internalEnergyR + velMagR * velMagR / 2.0 + pR / densityR;
        PetscReal HG_R = internalEnergyG_R + velMagR * velMagR / 2.0 + pR / densityG_R;
        PetscReal HL_R = internalEnergyL_R + velMagR * velMagR / 2.0 + pR / densityL_R;
        PetscReal HGL_R;
        if (alphaL > alphaR) {
            // liquid on right
            HGL_R = internalEnergyL_R + velMagR * velMagR / 2.0 + pR / densityL_R;
        } else if (alphaL < alphaR) {
            // gas on right
            HGL_R = internalEnergyG_R + velMagR * velMagR / 2.0 + pR / densityG_R;
        } else {
            // no discontinuous region
            HGL_R = 0.0;
        }
        //        flux[RHOE] = 0.5 * (HL + HR) * massFlux * areaMag;
        flux[CompressibleFlowFields::RHOE] = (0.5 * (HG_L + HG_R) * massFluxGG * areaMag * alphaMin) + (0.5 * (HGL_L + HGL_R) * massFluxGL * areaMag * alphaDif) +
                                             (0.5 * (HL_L + HL_R) * massFluxLL * areaMag * (1 - alphaMin - alphaDif));
        for (PetscInt n = 0; n < dim; n++) {
            //            flux[RHOU + n] = 0.5 * (velocityL[n] + velocityR[n]) * massFlux * areaMag + p12 * fg->normal[n];
            flux[CompressibleFlowFields::RHOU + n] =
                0.5 * (velocityL[n] + velocityR[n]) * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1 - alphaMin - alphaDif)) + p12 * fg->normal[n];
        }
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeVFFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x,
                                                                                                      const PetscScalar *fieldL, const PetscScalar *fieldR, const PetscScalar *gradL,
                                                                                                      const PetscScalar *gradR, const PetscInt *aOff, const PetscInt *aOff_x, const PetscScalar *auxL,
                                                                                                      const PetscScalar *auxR, const PetscScalar *gradAuxL, const PetscScalar *gradAuxR,
                                                                                                      PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    //     Decode left and right states
    PetscReal densityL;
    PetscReal densityG_L;
    PetscReal densityL_L;
    PetscReal normalVelocityL;  // uniform velocity in cell
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal internalEnergyG_L;
    PetscReal internalEnergyL_L;
    PetscReal aG_L;
    PetscReal aL_L;
    PetscReal MG_L;
    PetscReal ML_L;
    PetscReal pL;  // pressure equilibrium
    PetscReal tL;
    PetscReal alphaL;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(dim,
                                                              uOff,
                                                              fieldL,
                                                              norm,
                                                              &densityL,
                                                              &densityG_L,
                                                              &densityL_L,
                                                              &normalVelocityL,
                                                              velocityL,
                                                              &internalEnergyL,
                                                              &internalEnergyG_L,
                                                              &internalEnergyL_L,
                                                              &aG_L,
                                                              &aL_L,
                                                              &MG_L,
                                                              &ML_L,
                                                              &pL,
                                                              &tL,
                                                              &alphaL);

    PetscReal densityR;
    PetscReal densityG_R;
    PetscReal densityL_R;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal internalEnergyG_R;
    PetscReal internalEnergyL_R;
    PetscReal aG_R;
    PetscReal aL_R;
    PetscReal MG_R;
    PetscReal ML_R;
    PetscReal pR;
    PetscReal tR;
    PetscReal alphaR;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(dim,
                                                              uOff,
                                                              fieldR,
                                                              norm,
                                                              &densityR,
                                                              &densityG_R,
                                                              &densityL_R,
                                                              &normalVelocityR,
                                                              velocityR,
                                                              &internalEnergyR,
                                                              &internalEnergyG_R,
                                                              &internalEnergyL_R,
                                                              &aG_R,
                                                              &aL_R,
                                                              &MG_R,
                                                              &ML_R,
                                                              &pR,
                                                              &tR,
                                                              &alphaR);

    // get the face values
    PetscReal massFlux;
    PetscReal p12;
    // calculate gas sub-area of face (stratified flow model)
    //    PetscReal alpha = PetscMin(alphaR, alphaL);

    fluxCalculator::Direction directionG = twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFlux, &p12);
    if (directionG == fluxCalculator::LEFT) {
        flux[0] = massFlux * areaMag * alphaL;
    } else if (directionG == fluxCalculator::RIGHT) {
        flux[0] = massFlux * areaMag * alphaR;
    } else {
        flux[0] = massFlux * areaMag * 0.5 * (alphaL + alphaR);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::UpdateAuxVelocityField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                   const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[CompressibleFlowFields::RHO];

    for (PetscInt d = 0; d < dim; d++) {
        auxField[aOff[0] + d] = conservedValues[CompressibleFlowFields::RHOU + d] / density;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::UpdateAuxTemperatureField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                      const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;

    // For cell center, the norm is unity
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    PetscReal density;
    PetscReal densityG;
    PetscReal densityL;
    PetscReal normalVelocity;  // uniform velocity in cell
    PetscReal velocity[3];
    PetscReal internalEnergy;
    PetscReal internalEnergyG;
    PetscReal internalEnergyL;
    PetscReal aG;
    PetscReal aL;
    PetscReal MG;
    PetscReal ML;
    PetscReal p;  // pressure equilibrium
    PetscReal T;  // temperature equilibrium, Tg = TL
    PetscReal alpha;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(
        dim, uOff, conservedValues, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &T, &alpha);
    auxField[aOff[0]] = T;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::UpdateAuxPressureField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                   const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;

    // For cell center, the norm is unity
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    PetscReal density;
    PetscReal densityG;
    PetscReal densityL;
    PetscReal normalVelocity;  // uniform velocity in cell
    PetscReal velocity[3];
    PetscReal internalEnergy;
    PetscReal internalEnergyG;
    PetscReal internalEnergyL;
    PetscReal aG;
    PetscReal aL;
    PetscReal MG;
    PetscReal ML;
    PetscReal p;  // pressure equilibrium
    PetscReal T;  // temperature equilibrium, Tg = TL
    PetscReal alpha;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(
        dim, uOff, conservedValues, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &T, &alpha);
    auxField[aOff[0]] = p;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::UpdateAuxVolumeFractionField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                                         const PetscScalar *conservedValues, const PetscInt aOff[], PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;

    // For cell center, the norm is unity
    PetscReal norm[3];
    norm[0] = 1;
    norm[1] = 1;
    norm[2] = 1;

    PetscReal density;
    PetscReal densityG;
    PetscReal densityL;
    PetscReal normalVelocity;  // uniform velocity in cell
    PetscReal velocity[3];
    PetscReal internalEnergy;
    PetscReal internalEnergyG;
    PetscReal internalEnergyL;
    PetscReal aG;
    PetscReal aL;
    PetscReal MG;
    PetscReal ML;
    PetscReal p;  // pressure equilibrium
    PetscReal T;  // temperature equilibrium, Tg = TL
    PetscReal alpha;
    twoPhaseEulerAdvection->decoder->DecodeTwoPhaseEulerState(
        dim, uOff, conservedValues, norm, &density, &densityG, &densityL, &normalVelocity, velocity, &internalEnergy, &internalEnergyG, &internalEnergyL, &aG, &aL, &MG, &ML, &p, &T, &alpha);
    auxField[aOff[0]] = alpha;
    PetscFunctionReturn(0);
}

std::shared_ptr<ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseDecoder> ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CreateTwoPhaseDecoder(
    PetscInt dim, const std::shared_ptr<eos::EOS> eosGas, const std::shared_ptr<eos::EOS> eosLiquid) {
    // check if both perfect gases, use analytical solution
    auto perfectGasEos1 = std::dynamic_pointer_cast<eos::PerfectGas>(eosGas);
    auto perfectGasEos2 = std::dynamic_pointer_cast<eos::PerfectGas>(eosLiquid);
    // check if stiffened gas
    auto stiffenedGasEos1 = std::dynamic_pointer_cast<eos::StiffenedGas>(eosGas);
    auto stiffenedGasEos2 = std::dynamic_pointer_cast<eos::StiffenedGas>(eosLiquid);
    if (perfectGasEos1 && perfectGasEos2) {
        return std::make_shared<PerfectGasPerfectGasDecoder>(dim, perfectGasEos1, perfectGasEos2);
    } else if (perfectGasEos1 && stiffenedGasEos2) {
        return std::make_shared<PerfectGasStiffenedGasDecoder>(dim, perfectGasEos1, stiffenedGasEos2);
    }
    throw std::invalid_argument("Unknown combination of equation of states for ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseDecoder");
}

/**PerfectGasPerfectGasDecoder**************/
ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasPerfectGasDecoder::PerfectGasPerfectGasDecoder(PetscInt dim, std::shared_ptr<eos::PerfectGas> eosGas,
                                                                                                                  std::shared_ptr<eos::PerfectGas> eosLiquid)
    : eosGas(eosGas), eosLiquid(eosLiquid) {
    // Create the fake euler field
    auto fakeEulerField = ablate::domain::Field{.name = CompressibleFlowFields::EULER_FIELD, .numberComponents = 2 + dim, .offset = 0};

    // size up the scratch vars
    gasEulerFieldScratch.resize(2 + dim);
    liquidEulerFieldScratch.resize(2 + dim);

    // extract/store compute calls
    gasComputeTemperature = eosGas->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeEulerField});
    gasComputeInternalEnergy = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeEulerField});
    gasComputeSpeedOfSound = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeEulerField});
    gasComputePressure = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, {fakeEulerField});

    liquidComputeTemperature = eosLiquid->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeEulerField});
    liquidComputeInternalEnergy = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeEulerField});
    liquidComputeSpeedOfSound = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeEulerField});
    liquidComputePressure = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, {fakeEulerField});
}

void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasPerfectGasDecoder::DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                    const PetscReal *normal, PetscReal *density, PetscReal *densityG,
                                                                                                                    PetscReal *densityL, PetscReal *normalVelocity, PetscReal *velocity,
                                                                                                                    PetscReal *internalEnergy, PetscReal *internalEnergyG, PetscReal *internalEnergyL,
                                                                                                                    PetscReal *aG, PetscReal *aL, PetscReal *MG, PetscReal *ML, PetscReal *p,
                                                                                                                    PetscReal *T, PetscReal *alpha) {
    // (RHO, RHOE, RHOU, RHOV, RHOW)
    const int EULER_FIELD = 1;
    const int VF_FIELD = 0;
    // decode
    *density = conservedValues[CompressibleFlowFields::RHO + uOff[EULER_FIELD]];
    PetscReal totalEnergy = conservedValues[CompressibleFlowFields::RHOE + uOff[EULER_FIELD]] / (*density);
    PetscReal densityVF = conservedValues[uOff[VF_FIELD]];

    // Get the velocity in this direction, and kinetic energy
    (*normalVelocity) = 0.0;
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[CompressibleFlowFields::RHOU + d + uOff[EULER_FIELD]] / (*density);
        (*normalVelocity) += velocity[d] * normal[d];
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;
    (*internalEnergy) = (totalEnergy)-ke;

    // mass fractions
    PetscReal Yg = densityVF / (*density);
    PetscReal Yl = ((*density) - densityVF) / (*density);

    PetscReal R1 = eosGas->GetGasConstant();
    PetscReal R2 = eosLiquid->GetGasConstant();
    PetscReal gamma1 = eosGas->GetSpecificHeatRatio();
    PetscReal gamma2 = eosLiquid->GetSpecificHeatRatio();
    PetscReal cv1 = R1 / (gamma1 - 1);
    PetscReal cv2 = R2 / (gamma2 - 1);

    PetscReal eG = (*internalEnergy) / (Yg + Yl * cv2 / cv1);
    PetscReal etG = eG + ke;
    PetscReal eL = cv2 / cv1 * eG;

    PetscReal etL = eL + ke;
    PetscReal rhoG = (*density) * (Yg + Yl * eL / eG * (gamma2 - 1) / (gamma1 - 1));
    PetscReal rhoL = rhoG * eG / eL * (gamma1 - 1) / (gamma2 - 1);

    PetscReal pG = 0;
    PetscReal pL;
    PetscReal a1 = 0;
    PetscReal a2 = 0;

    // Fill the scratch array for gas
    liquidEulerFieldScratch[CompressibleFlowFields::RHO] = rhoL;
    liquidEulerFieldScratch[CompressibleFlowFields::RHOE] = rhoL * etL;
    for (PetscInt d = 0; d < dim; d++) {
        liquidEulerFieldScratch[CompressibleFlowFields::RHOU + d] = velocity[d] * rhoL;
    }

    // Decode the gas
    {
        liquidComputeTemperature.function(liquidEulerFieldScratch.data(), T, liquidComputeTemperature.context.get()) >> checkError;
        liquidComputeInternalEnergy.function(liquidEulerFieldScratch.data(), *T, &eL, liquidComputeInternalEnergy.context.get()) >> checkError;
        liquidComputeSpeedOfSound.function(liquidEulerFieldScratch.data(), *T, &a2, liquidComputeSpeedOfSound.context.get()) >> checkError;
        liquidComputePressure.function(liquidEulerFieldScratch.data(), *T, &pL, liquidComputePressure.context.get()) >> checkError;
    }

    // Fill the scratch array for gas
    gasEulerFieldScratch[CompressibleFlowFields::RHO] = rhoG;
    gasEulerFieldScratch[CompressibleFlowFields::RHOE] = rhoG * etG;
    for (PetscInt d = 0; d < dim; d++) {
        gasEulerFieldScratch[CompressibleFlowFields::RHOU + d] = velocity[d] * rhoG;
    }

    // Decode the gas
    {
        gasComputeTemperature.function(gasEulerFieldScratch.data(), T, gasComputeTemperature.context.get()) >> checkError;
        gasComputeInternalEnergy.function(gasEulerFieldScratch.data(), *T, &eG, gasComputeInternalEnergy.context.get()) >> checkError;
        gasComputeSpeedOfSound.function(gasEulerFieldScratch.data(), *T, &a1, gasComputeSpeedOfSound.context.get()) >> checkError;
        gasComputePressure.function(gasEulerFieldScratch.data(), *T, &pG, gasComputePressure.context.get()) >> checkError;
    }

    // once state defined
    *densityG = rhoG;
    *densityL = rhoL;
    *internalEnergyG = eG;
    *internalEnergyL = eL;
    *alpha = densityVF / (*densityG);
    *p = pG;  // pressure equilibrium, pG = pL
    *aG = a1;
    *aL = a2;
    *MG = (*normalVelocity) / (*aG);
    *ML = (*normalVelocity) / (*aL);
}

/**PerfectGasStiffenedGasDecoder**************/
ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasStiffenedGasDecoder::PerfectGasStiffenedGasDecoder(PetscInt dim, std::shared_ptr<eos::PerfectGas> eosGas,
                                                                                                                      std::shared_ptr<eos::StiffenedGas> eosLiquid)
    : eosGas(eosGas), eosLiquid(eosLiquid) {
    // Create the fake euler field
    auto fakeEulerField = ablate::domain::Field{.name = CompressibleFlowFields::EULER_FIELD, .numberComponents = 2 + dim, .offset = 0};

    // size up the scratch vars
    gasEulerFieldScratch.resize(2 + dim);
    liquidEulerFieldScratch.resize(2 + dim);

    // extract/store compute calls
    gasComputeTemperature = eosGas->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeEulerField});
    gasComputeInternalEnergy = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeEulerField});
    gasComputeSpeedOfSound = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeEulerField});
    gasComputePressure = eosGas->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, {fakeEulerField});

    liquidComputeTemperature = eosLiquid->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, {fakeEulerField});
    liquidComputeInternalEnergy = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, {fakeEulerField});
    liquidComputeSpeedOfSound = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::SpeedOfSound, {fakeEulerField});
    liquidComputePressure = eosLiquid->GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty::Pressure, {fakeEulerField});
}
void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasStiffenedGasDecoder::DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
                                                                                                                      const PetscReal *normal, PetscReal *density, PetscReal *densityG,
                                                                                                                      PetscReal *densityL, PetscReal *normalVelocity, PetscReal *velocity,
                                                                                                                      PetscReal *internalEnergy, PetscReal *internalEnergyG, PetscReal *internalEnergyL,
                                                                                                                      PetscReal *aG, PetscReal *aL, PetscReal *MG, PetscReal *ML, PetscReal *p,
                                                                                                                      PetscReal *T, PetscReal *alpha) {
    const int EULER_FIELD = 1;
    const int VF_FIELD = 0;

    // decode
    *density = conservedValues[CompressibleFlowFields::RHO + uOff[EULER_FIELD]];
    PetscReal totalEnergy = conservedValues[CompressibleFlowFields::RHOE + uOff[EULER_FIELD]] / (*density);
    PetscReal densityVF = conservedValues[uOff[VF_FIELD]];

    // Get the velocity in this direction, and kinetic energy
    (*normalVelocity) = 0.0;
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[CompressibleFlowFields::RHOU + d + uOff[EULER_FIELD]] / (*density);
        (*normalVelocity) += velocity[d] * normal[d];
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;
    (*internalEnergy) = (totalEnergy)-ke;

    // mass fractions
    PetscReal Yg = densityVF / (*density);
    PetscReal Yl = ((*density) - densityVF) / (*density);

    PetscReal R1 = eosGas->GetGasConstant();
    PetscReal cp2 = eosLiquid->GetSpecificHeatCp();
    PetscReal p02 = eosLiquid->GetReferencePressure();
    PetscReal gamma1 = eosGas->GetSpecificHeatRatio();
    PetscReal gamma2 = eosLiquid->GetSpecificHeatRatio();
    PetscReal cv1 = R1 / (gamma1 - 1);

    PetscReal etot = (*internalEnergy);
    PetscReal A = cp2 / cv1 / gamma2;
    PetscReal B = Yg + Yl * A;
    PetscReal D = p02 / (*density) - etot;
    PetscReal E = Yg * p02 / (*density) + Yl * A * etot;
    PetscReal eG, eL, a, b, c, root1, root2;
    if (Yg < 1E-3) {  // avoid divide by zero
        a = B * (Yg * (gamma2 - 1) - Yg * (gamma1 - 1) - gamma2 * B);
        b = etot * Yg * (gamma1 - 1) + etot * B + Yg * (gamma2 - 1) * D - gamma2 * D * B;
        c = etot * D;
        if (PetscAbs(a) < 1E-5) {
            eG = -c / b;
        } else {
            root1 = (-b + PetscSqrtReal(PetscSqr(b) - 4 * a * c)) / (2 * a);
            root2 = (-b - PetscSqrtReal(PetscSqr(b) - 4 * a * c)) / (2 * a);
            if (root1 > 1E-5 && root2 > 1E-5) {
                eG = PetscMin(root1, root2);
            } else {
                eG = PetscMax(root1, root2);  // take positive root
                if (eG < 0) {                 // negative internal energy not physical
                    throw std::invalid_argument("ablate::finiteVolume::twoPhaseEulerAdvection PerfectGas/StiffenedGas DecodeState cannot result in negative internal energy eG");
                }
            }
        }

        eL = ((*internalEnergy) - Yg * eG) / Yl;
        if (eL < 0) {
            throw std::invalid_argument("ablate::finiteVolume::twoPhaseEulerAdvection PerfectGas/StiffenedGas DecodeState cannot result in negative internal energy eL");
        }
    } else {  // else if Yl<10e-5,
        a = B * Yl * (Yg * (gamma2 - 1) - gamma2 * B - Yg * (gamma1 - 1));
        b = Yg * (gamma1 - 1) * B * etot + Yg * (gamma1 - 1) * Yl * A * etot - Yg * (gamma2 - 1) * E + gamma2 * E * B + gamma2 * Yl * B * A * etot;
        c = (-A) * etot * (Yg * (gamma1 - 1) * etot + gamma2 * E);
        if (PetscAbs(a) < 1E-5) {
            eL = -c / b;
        } else {
            root1 = (-b + PetscSqrtReal(PetscSqr(b) - 4 * a * c)) / (2 * a);
            root2 = (-b - PetscSqrtReal(PetscSqr(b) - 4 * a * c)) / (2 * a);
            if (root1 > 1E-5 && root2 > 1E-5) {
                eL = PetscMin(root1, root2);
            } else {
                eL = PetscMax(root1, root2);  // take positive root
                if (eL < 0) {                 // negative internal energy not physical
                    throw std::invalid_argument("ablate::finiteVolume::twoPhaseEulerAdvection PerfectGas/StiffenedGas DecodeState cannot result in negative internal energy eL");
                }
            }
        }
        eG = ((*internalEnergy) - Yl * eL) / Yg;
        if (eG < 0) {
            throw std::invalid_argument("ablate::finiteVolume::twoPhaseEulerAdvection PerfectGas/StiffenedGas DecodeState cannot result in negative internal energy eG");
        }
    }

    PetscReal etG = eG + ke;
    PetscReal etL = eL + ke;
    PetscReal ar = -(gamma2 - 1) * eL;
    PetscReal br = (gamma1 - 1) * Yg * (*density) * eG + (gamma2 - 1) * eL * (*density) * Yl + gamma2 * p02;
    PetscReal cr = -gamma2 * p02 * Yl * (*density);
    PetscReal root1r = (-br + PetscSqrtReal(PetscSqr(br) - 4 * ar * cr)) / (2 * ar);
    PetscReal root2r = (-br - PetscSqrtReal(PetscSqr(br) - 4 * ar * cr)) / (2 * ar);
    PetscReal rhoL;
    if (PetscAbs(ar) < 1E-5) {
        rhoL = -cr / br;
    } else {
        rhoL = PetscMax(root1r, root2r);
        if (rhoL < 1E-5) {  // negative density not physical
            throw std::invalid_argument("ablate::finiteVolume::twoPhaseEulerAdvection PerfectGas/StiffenedGas DecodeState cannot result in negative density rhoL");
        }
    }
    PetscReal rhoG = ((gamma2 - 1) * eL * rhoL - gamma2 * p02) / (gamma1 - 1) / eG;
    if (rhoG < 1E-5) {  // negative density not physical
        throw std::invalid_argument("ablate::finiteVolume::twoPhaseEulerAdvection PerfectGas/StiffenedGas DecodeState cannot result in negative density rhoG");
    }
    PetscReal pG = 0;
    PetscReal pL;
    PetscReal a1 = 0;
    PetscReal a2 = 0;

    // Fill the scratch array for gas
    liquidEulerFieldScratch[CompressibleFlowFields::RHO] = rhoL;
    liquidEulerFieldScratch[CompressibleFlowFields::RHOE] = rhoL * etL;
    for (PetscInt d = 0; d < dim; d++) {
        liquidEulerFieldScratch[CompressibleFlowFields::RHOU + d] = velocity[d] * rhoL;
    }

    // Decode the gas
    {
        liquidComputeTemperature.function(liquidEulerFieldScratch.data(), T, liquidComputeTemperature.context.get()) >> checkError;
        liquidComputeInternalEnergy.function(liquidEulerFieldScratch.data(), *T, &eL, liquidComputeInternalEnergy.context.get()) >> checkError;
        liquidComputeSpeedOfSound.function(liquidEulerFieldScratch.data(), *T, &a2, liquidComputeSpeedOfSound.context.get()) >> checkError;
        liquidComputePressure.function(liquidEulerFieldScratch.data(), *T, &pL, liquidComputePressure.context.get()) >> checkError;
    }

    // Fill the scratch array for gas
    gasEulerFieldScratch[CompressibleFlowFields::RHO] = rhoG;
    gasEulerFieldScratch[CompressibleFlowFields::RHOE] = rhoG * etG;
    for (PetscInt d = 0; d < dim; d++) {
        gasEulerFieldScratch[CompressibleFlowFields::RHOU + d] = velocity[d] * rhoG;
    }

    // Decode the gas
    {
        gasComputeTemperature.function(gasEulerFieldScratch.data(), T, gasComputeTemperature.context.get()) >> checkError;
        gasComputeInternalEnergy.function(gasEulerFieldScratch.data(), *T, &eG, gasComputeInternalEnergy.context.get()) >> checkError;
        gasComputeSpeedOfSound.function(gasEulerFieldScratch.data(), *T, &a1, gasComputeSpeedOfSound.context.get()) >> checkError;
        gasComputePressure.function(gasEulerFieldScratch.data(), *T, &pG, gasComputePressure.context.get()) >> checkError;
    }

    // once state defined
    *densityG = rhoG;
    *densityL = rhoL;
    *internalEnergyG = eG;
    *internalEnergyL = eL;
    *alpha = densityVF / (*densityG);
    *p = pG;  // pressure equilibrium, pG = pL
    *aG = a1;
    *aL = a2;
    *MG = (*normalVelocity) / (*aG);
    *ML = (*normalVelocity) / (*aL);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::TwoPhaseEulerAdvection, "", ARG(ablate::eos::EOS, "eosGas", ""), ARG(ablate::eos::EOS, "eosLiquid", ""),
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorGasGas", ""), ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorGasLiquid", ""),
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorLiquidGas", ""), ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorLiquidLiquid", ""));

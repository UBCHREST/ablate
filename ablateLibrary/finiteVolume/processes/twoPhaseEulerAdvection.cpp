#include "twoPhaseEulerAdvection.hpp"

static inline void NormVector(PetscInt dim, const PetscReal* in, PetscReal* out) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d = 0; d < dim; d++) {
        out[d] = in[d] / mag;
    }
}

static inline PetscReal MagVector(PetscInt dim, const PetscReal* in) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    return PetscSqrtReal(mag);
}

void ablate::flow::processes::TwoPhaseEulerAdvection::DecodeTwoPhaseEulerState(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid, PetscInt dim, const PetscReal *conservedValues,
                                                                               const PetscReal *normal, PetscReal *density, PetscReal *densityG, PetscReal *densityL,
                                                                               PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, PetscReal *internalEnergyG,
                                                                               PetscReal *internalEnergyL, PetscReal *aG, PetscReal *aL, PetscReal *MG, PetscReal *ML, PetscReal *p, PetscReal *alpha) {
    const int EULER_FIELD = 1; // denstiyVF is [0] field
    // (densityVF, RHO, RHOE, RHOU, RHOV, RHOW)
    // decode
    PetscReal densityVF = conservedValues[0];
    *density = conservedValues[0 + EULER_FIELD];
    PetscReal totalEnergy = conservedValues[1 + EULER_FIELD]/(*density);

    // Get the velocity in this direction, and kinetic energy
    (*normalVelocity) = 0.0;
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[2 + EULER_FIELD + d] / (*density);
        (*normalVelocity) += velocity[d] * normal[d];
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;
    (*internalEnergy) = (totalEnergy)-ke;

    // Get mass fractions
    PetscReal Yg = densityVF / (*density);
    PetscReal Yl = ((*density) - densityVF) / (*density);

    PetscReal pG;
    PetscReal pL;
    // additional equations:
    // 1/density = Yg/densityG + Yl/densityL;
    // internalEnergy = Yg*internalEnergyG + Yl*internalEnergyL;
    PetscReal eG = (*internalEnergy);
    PetscReal etG = eG + ke;
    PetscReal eL = ((*internalEnergy) - Yg*eG)/Yl;
    PetscReal etL = eL + ke;
    PetscReal rhoG = Yg * (*density);
    PetscReal rhoL = Yl / (1/(*density) - Yg/rhoG);
    // while (PetscAbs(delp) > PerrorTol){
    // guess new eG, rhoG; calculate new eL, rhoL
    // decode the state in the eos
    eosGas->GetDecodeStateFunction()(dim, rhoG, etG, velocity, NULL, internalEnergy, aG, &pG, eosGas->GetDecodeStateContext());
    eosLiquid->GetDecodeStateFunction()(dim, rhoL, etL, velocity, NULL, internalEnergy, aL, &pL, eosLiquid->GetDecodeStateContext());
    // guess et1, rho1; return e1, a1, p1 | guess et2, rho2; return e2, a2, p2
    // compare pG, pL
    //PetscReal delp = pG - pL
    // if delp >0; choose new eG, rhoG this way
    // elseif delp <0: choose new eG, rhoG some other way
    // iterate result: densityG, densityL, internalEnergyG, internalEnergyL, p, T

    // once state defined
    *densityG = rhoG;
    *densityL = rhoL;
    *internalEnergyG = eG;
    *internalEnergyL = eL;
//    *p = pG;
    *MG = (*normalVelocity) / (*aG);
    *ML = (*normalVelocity) / (*aL);
    *alpha = densityVF / (*densityG);
}

ablate::flow::processes::TwoPhaseEulerAdvection::TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid,
                                                                        std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas,
                                                                        std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid,
                                                                        std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid)
    : eosGas(eosGas), eosLiquid(eosLiquid), fluxCalculatorGasGas(fluxCalculatorGasGas), fluxCalculatorGasLiquid(fluxCalculatorGasLiquid), fluxCalculatorLiquidLiquid(fluxCalculatorLiquidLiquid) {}
void ablate::flow::processes::TwoPhaseEulerAdvection::Initialize(ablate::flow::FVFlow& flow) {
    // Currently no option for species advection
    flow.RegisterRHSFunction(CompressibleFlowComputeEulerFlux, this, "euler", {"densityVF","euler"},{});
    flow.RegisterRHSFunction(CompressibleFlowComputeVFFlux, this, "densityVF", {"densityVF","euler"},{});
}
PetscErrorCode ablate::flow::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x,
                                                                                                 const PetscScalar *fieldL, const PetscScalar *fieldR, const PetscScalar *gradL,
                                                                                                 const PetscScalar *gradR, const PetscInt *aOff, const PetscInt *aOff_x, const PetscScalar *auxL,
                                                                                                 const PetscScalar *auxR, const PetscScalar *gradAuxL, const PetscScalar *gradAuxR, PetscScalar *flux,
                                                                                                 void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;
    const int EULER_FIELD = 1;  // densityVF is [0] field
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
    PetscReal alphaL;
    DecodeTwoPhaseEulerState(twoPhaseEulerAdvection->eosGas,twoPhaseEulerAdvection->eosLiquid, dim, fieldL + uOff[EULER_FIELD], norm, &densityL, &densityG_L, &densityL_L, &normalVelocityL, velocityL, &internalEnergyL, &internalEnergyG_L, &internalEnergyL_L, &aG_L, &aL_L, &MG_L, &ML_L, &pL, &alphaL);

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
    PetscReal alphaR;
    DecodeTwoPhaseEulerState(twoPhaseEulerAdvection->eosGas,twoPhaseEulerAdvection->eosLiquid, dim, fieldR + uOff[EULER_FIELD], norm, &densityR, &densityG_R, &densityL_R, &normalVelocityR, velocityR, &internalEnergyR, &internalEnergyG_R, &internalEnergyL_R, &aG_R, &aL_R, &MG_R, &ML_R, &pR, &alphaR);

    // get the face values
    PetscReal massFluxGG;
    PetscReal massFluxGL;
    PetscReal massFluxLL;
    PetscReal p12;  // pressure equilibrium ( might need to make sure they match)

    // call flux calculator 3 times, gas-gas, gas-liquid, liquid-liquid regions
    fluxCalculator::Direction directionG =
        twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFluxGG, &p12);
//    fluxCalculator::Direction directionL =  // should be same direction, if not, big problem
        twoPhaseEulerAdvection->fluxCalculatorLiquidLiquid->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorLiquidLiquid->GetFluxCalculatorContext(), normalVelocityL, aL_L, densityL_L, pL, normalVelocityR, aL_R, densityL_R, pR, &massFluxLL, &p12);
    if (alphaL > alphaR) {
        // gas on left, liquid on right
            twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorFunction()(twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aL_R, densityL_R, pR, &massFluxGL, &p12);
    } else if (alphaL < alphaR) {
        // liquid on left, gas on right
            twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorFunction()(twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorContext(), normalVelocityL, aL_L, densityL_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFluxGL, &p12);
    } else {
        // no discontinuous region
        massFluxGL=0.0;
    }

    // Calculate total flux
    PetscReal alphaMin = PetscMin(alphaR,alphaL);
    PetscReal alphaDif = PetscAbs(alphaL - alphaR);
    if (directionG == fluxCalculator::LEFT) { // direction of GG,LL,LG should match since uniform velocity???
//        flux[RHO] = massFlux * areaMag;
        flux[0 + EULER_FIELD] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * (1-alphaMin));
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
        }else {
            // no discontinuous region
            HGL_L=0.0;
        }
//        flux[RHOE] = HL * massFlux * areaMag;
        flux[1 + EULER_FIELD] = (HG_L * massFluxGG * areaMag * alphaMin) + (HGL_L * massFluxGL * areaMag * alphaDif) + (HL_L * massFluxLL * areaMag * (1-alphaMin));
        for (PetscInt n = 0; n < dim; n++) {
//            flux[RHOU + n] = velocityL[n] * massFlux * areaMag + p12 * fg->normal[n];
            flux[2 + EULER_FIELD + n] = velocityL[n] * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1-alphaMin)) + p12 * fg->normal[n];
        }
    } else if (directionG == fluxCalculator::RIGHT) {
//        flux[RHO] = massFlux * areaMag;
        flux[0 + EULER_FIELD] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * (1-alphaMin));
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
        }else {
            // no discontinuous region
            HGL_R=0.0;
        }
//        flux[RHOE] = HR * massFlux * areaMag;
        flux[1 + EULER_FIELD] = (HG_R * massFluxGG * areaMag * alphaMin) + (HGL_R * massFluxGL * areaMag * alphaDif) + (HL_R * massFluxLL * areaMag * (1-alphaMin));
        for (PetscInt n = 0; n < dim; n++) {
//            flux[RHOU + n] = velocityR[n] * massFlux * areaMag + p12 * fg->normal[n];
            flux[2 + EULER_FIELD + n] = velocityR[n] * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1-alphaMin)) + p12 * fg->normal[n];
        }
    } else {
//        flux[RHO] = massFlux * areaMag;
        flux[0 + EULER_FIELD] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * (1-alphaMin));

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
        }else {
            // no discontinuous region
            HGL_L=0.0;
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
        }else {
            // no discontinuous region
            HGL_R=0.0;
        }
//        flux[RHOE] = 0.5 * (HL + HR) * massFlux * areaMag;
        flux[1 + EULER_FIELD] = (0.5*(HG_L+HG_R) * massFluxGG * areaMag * alphaMin) + (0.5*(HGL_L+HGL_R) * massFluxGL * areaMag * alphaDif) + (0.5*(HL_L+HL_R) * massFluxLL * areaMag * (1-alphaMin));

        for (PetscInt n = 0; n < dim; n++) {
//            flux[RHOU + n] = 0.5 * (velocityL[n] + velocityR[n]) * massFlux * areaMag + p12 * fg->normal[n];
            flux[2 + EULER_FIELD + n] = 0.5*(velocityL[n]+velocityR[n]) * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1-alphaMin)) + p12 * fg->normal[n];
        }
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::flow::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeVFFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x,
                                                                                              const PetscScalar *fieldL, const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR,
                                                                                              const PetscInt *aOff, const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR,
                                                                                              const PetscScalar *gradAuxL, const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection*)ctx;
    const int VF_FIELD = 0;
    const int EULER_FIELD = 1;

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
    PetscReal alphaL;
    DecodeTwoPhaseEulerState(twoPhaseEulerAdvection->eosGas,twoPhaseEulerAdvection->eosLiquid, dim, fieldR + uOff[EULER_FIELD], norm, &densityL, &densityG_L, &densityL_L, &normalVelocityL, velocityL, &internalEnergyL, &internalEnergyG_L, &internalEnergyL_L, &aG_L, &aL_L, &MG_L, &ML_L, &pL, &alphaL);

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
    PetscReal alphaR;
    DecodeTwoPhaseEulerState(twoPhaseEulerAdvection->eosGas,twoPhaseEulerAdvection->eosLiquid, dim, fieldR + uOff[EULER_FIELD], norm, &densityR, &densityG_R, &densityL_R, &normalVelocityR, velocityR, &internalEnergyR, &internalEnergyG_R, &internalEnergyL_R, &aG_R, &aL_R, &MG_R, &ML_R, &pR, &alphaR);

    // get the face values
    PetscReal massFlux;
    PetscReal p12;
    // calculate gas sub-area of face (stratified flow model)
    PetscReal alpha = PetscMin(alphaR,alphaL);

    // how do I check that densityVF/dt is being integrated over Volume*alpha????? (can I get this to cancel out??)
    twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFlux, &p12);
    flux[0] = massFlux * areaMag * alpha; // is this correct???


    PetscFunctionReturn(0);
}


#include "parser/registrar.hpp"
REGISTER(ablate::flow::processes::FlowProcess, ablate::flow::processes::TwoPhaseEulerAdvection, "",
         ARG(ablate::eos::EOS,"eosGas",""),ARG(ablate::eos::EOS,"eosLiquid",""),
         ARG(ablate::flow::fluxCalculator::FluxCalculator,"fluxCalculatorGasGas",""),ARG(ablate::flow::fluxCalculator::FluxCalculator,"fluxCalculatorGasLiquid",""),ARG(ablate::flow::fluxCalculator::FluxCalculator,"fluxCalculatorLiquidLiquid",""));

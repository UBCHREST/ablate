#include "twoPhaseEulerAdvection.hpp"

#include <utility>
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

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::FormFunctionGas(SNES snes, Vec x, Vec F, void *ctx){
    auto decodeDataStruct = (DecodeDataStructGas *)ctx;
    const PetscReal *ax;
    PetscReal *aF;
    VecGetArrayRead(x, &ax);
    // ax = [rhog, rhol, eg, el]
    PetscReal rhoG = ax[0];
    PetscReal rhoL = ax[1];
    PetscReal eG = ax[2];
    PetscReal eL = ax[3];

    PetscReal gamma1 = decodeDataStruct->gam1;
    PetscReal gamma2 = decodeDataStruct->gam2;
    PetscReal Y1 = decodeDataStruct->Yg;
    PetscReal Y2 = decodeDataStruct->Yl;
    PetscReal rho = decodeDataStruct->rhotot;
    PetscReal e = decodeDataStruct->etot;
    PetscReal cv1 = decodeDataStruct->cvg;
    PetscReal cp2 = decodeDataStruct->cpl;
    PetscReal p02 = decodeDataStruct->p0l;

    VecGetArray(F, &aF);
    aF[0] = (gamma1 - 1)*eG*rhoG - (gamma2 - 1)*eL*rhoL + gamma2*p02; // pG - pL = 0, pressure equilibrium
    aF[1] = eG*rhoL/cv1 - gamma2/cp2*(eL*rhoL - p02); // TG - TL = 0, temperature equilibrium
    aF[2] = Y1*rho*rhoL + Y2*rho*rhoG - rhoG*rhoL;
    aF[3] = Y1*eG + Y2*eL - e;

    VecRestoreArrayRead(x, &ax);
    VecRestoreArray(F, &aF);
    return 0;

}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::FormJacobianGas(SNES snes, Vec x, Mat J, Mat P, void *ctx){
    auto decodeDataStruct = (DecodeDataStructGas *)ctx;
    const PetscReal *ax;
    PetscReal v[16];
    PetscInt row[4] = {0, 1, 2, 3}, col[4] = {0, 1, 2, 3};
    VecGetArrayRead(x, &ax);
    // ax = [rhog, rhol, eg, el]
    PetscReal rhoG = ax[0];
    PetscReal rhoL = ax[1];
    PetscReal eG = ax[2];
    PetscReal eL = ax[3];

    PetscReal gamma1 = decodeDataStruct->gam1;
    PetscReal gamma2 = decodeDataStruct->gam2;
    PetscReal Y1 = decodeDataStruct->Yg;
    PetscReal Y2 = decodeDataStruct->Yl;
    PetscReal rho = decodeDataStruct->rhotot;
//    PetscReal e = decodeDataStruct->etot;
    PetscReal cv1 = decodeDataStruct->cvg;
    PetscReal cp2 = decodeDataStruct->cpl;
//    PetscReal p02 = decodeDataStruct->p0l;

    v[0] = (gamma1-1)*eG; v[1] = -(gamma2-1)*eL; v[2] = (gamma1-1)*rhoG; v[3] = -(gamma2-1)*rhoL;
    v[4] = 0.0; v[5] = eG/cv1 - gamma2*eG/cp2; v[6] = rhoL/cv1; v[7] = -gamma2*rhoL/cp2;
    v[8] = Y2*rho-rhoL; v[9] = Y1*rho-rhoG; v[10] = 0.0; v[11] = 0.0;
    v[12] = 0.0; v[13] = 0.0; v[14] = Y1; v[15] = Y2;
    VecRestoreArrayRead(x, &ax);
    MatSetValues(P,4,row,4,col,v,INSERT_VALUES);
    MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);
    if (J!=P){
        MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
    }
    return 0;

}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::FormFunctionStiff(SNES snes, Vec x, Vec F, void *ctx){
    auto decodeDataStruct = (DecodeDataStructStiff *)ctx;
    const PetscReal *ax;
    PetscReal *aF;
    VecGetArrayRead(x, &ax);
    // ax = [rhog, rhol, eg, el]
    PetscReal rhoG = ax[0];
    PetscReal rhoL = ax[1];
    PetscReal eG = ax[2];
    PetscReal eL = ax[3];

    PetscReal gamma1 = decodeDataStruct->gam1;
    PetscReal gamma2 = decodeDataStruct->gam2;
    PetscReal Y1 = decodeDataStruct->Yg;
    PetscReal Y2 = decodeDataStruct->Yl;
    PetscReal rho = decodeDataStruct->rhotot;
    PetscReal e = decodeDataStruct->etot;
    PetscReal cp1 = decodeDataStruct->cpg;
    PetscReal cp2 = decodeDataStruct->cpl;
    PetscReal p01 = decodeDataStruct->p0g;
    PetscReal p02 = decodeDataStruct->p0l;

    VecGetArray(F, &aF);
    aF[0] = (gamma1 - 1)*eG*rhoG - gamma1*p01 - (gamma2 - 1)*eL*rhoL + gamma2*p02; // pG - pL = 0, pressure equilibrium
    aF[1] = gamma1/cp1*rhoL*(eG*rhoG-p01) - gamma2/cp2*rhoG*(eL*rhoL - p02); // TG - TL = 0, temperature equilibrium
    aF[2] = Y1*rho*rhoL + Y2*rho*rhoG - rhoG*rhoL;
    aF[3] = Y1*eG + Y2*eL - e;

    VecRestoreArrayRead(x, &ax);
    VecRestoreArray(F, &aF);
    return 0;

}

PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::FormJacobianStiff(SNES snes, Vec x, Mat J, Mat P, void *ctx){
    auto decodeDataStruct = (DecodeDataStructStiff *)ctx;
    const PetscReal *ax;
    PetscReal v[16];
    PetscInt row[4] = {0, 1, 2, 3}, col[4] = {0, 1, 2, 3};
    VecGetArrayRead(x, &ax);
    // ax = [rhog, rhol, eg, el]
    PetscReal rhoG = ax[0];
    PetscReal rhoL = ax[1];
    PetscReal eG = ax[2];
    PetscReal eL = ax[3];

    PetscReal gamma1 = decodeDataStruct->gam1;
    PetscReal gamma2 = decodeDataStruct->gam2;
    PetscReal Y1 = decodeDataStruct->Yg;
    PetscReal Y2 = decodeDataStruct->Yl;
    PetscReal rho = decodeDataStruct->rhotot;
//    PetscReal e = decodeDataStruct->etot;
    PetscReal cp1 = decodeDataStruct->cpg;
    PetscReal cp2 = decodeDataStruct->cpl;
    PetscReal p01 = decodeDataStruct->p0g;
    PetscReal p02 = decodeDataStruct->p0l;

    // need to check Jacobian, not getting correct solution
    v[0] = (gamma1-1)*eG; v[1] = -(gamma2-1)*eL; v[2] = (gamma1-1)*rhoG; v[3] = -(gamma2-1)*rhoL;
    v[4] = gamma1/cp1*eG*rhoL - gamma2/cp2*eL*rhoL + gamma2/cp2*p02; v[5] = gamma1/cp1*eG*rhoG - gamma1/cp1*p01 - gamma2/cp2*eL*rhoG; v[6] = gamma1/cp1*rhoG*rhoL; v[7] = -gamma2/cp2*rhoG*rhoL;
    v[8] = Y2*rho-rhoL; v[9] = Y1*rho-rhoG; v[10] = 0.0; v[11] = 0.0;
    v[12] = 0.0; v[13] = 0.0; v[14] = Y1; v[15] = Y2;
    VecRestoreArrayRead(x, &ax);
    MatSetValues(P,4,row,4,col,v,INSERT_VALUES);
    MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);
    if (J!=P){
        MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
    }
    return 0;

}

ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidGas,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid)
    : eosGas(std::move(eosGas)),
      eosLiquid(std::move(eosLiquid)),
      fluxCalculatorGasGas(std::move(fluxCalculatorGasGas)),
      fluxCalculatorGasLiquid(std::move(fluxCalculatorGasLiquid)),
      fluxCalculatorLiquidGas(std::move(fluxCalculatorLiquidGas)),
      fluxCalculatorLiquidLiquid(std::move(fluxCalculatorLiquidLiquid)) {}

void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
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
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL,
                                                                                                         const PetscScalar *fieldR, const PetscInt *aOff, const PetscScalar *auxL,
                                                                                                         const PetscScalar *auxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection *)ctx;
    // Compute the norm of cell face
    PetscReal norm[3], newnormal[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);
    for (PetscInt n = 0; n < dim; n++) {
        if (PetscAbs(fg->normal[n]) < 1e-12){
            newnormal[n]=0.0;
        } else{
            newnormal[n]=round(fg->normal[n]*1e12)/1e12;
        }
    }

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
    PetscReal p12GG;  // interface pressure
    PetscReal p12GL;
    PetscReal p12LL;

    // call flux calculator 3 times, gas-gas, gas-liquid, liquid-liquid regions
    fluxCalculator::Direction directionG = twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFluxGG, &p12GG);
    // should be same direction, if not, big problem
    PetscReal alphaMin = PetscMin(alphaR, alphaL);
    PetscReal alphaDif = PetscAbs(alphaL - alphaR);
    PetscReal alphaLiq;
    fluxCalculator::Direction directionL;
        if ( (alphaMin+alphaDif) >= (1.0-1e-12) ){
        alphaLiq = 0.0;
        massFluxLL = 0.0;
        p12LL = p12GG;
    } else{
        alphaLiq = 1 - alphaMin - alphaDif;
        directionL = twoPhaseEulerAdvection->fluxCalculatorLiquidLiquid->GetFluxCalculatorFunction()(
            twoPhaseEulerAdvection->fluxCalculatorLiquidLiquid->GetFluxCalculatorContext(), normalVelocityL, aL_L, densityL_L, pL, normalVelocityR, aL_R, densityL_R, pR, &massFluxLL, &p12LL);
    }
    fluxCalculator::Direction directionGL;
    if (alphaL > alphaR) {
        // gas on left, liquid on right
        directionGL = twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorFunction()(
            twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aL_R, densityL_R, pR, &massFluxGL, &p12GL);
    } else if (alphaL < alphaR) {
        // liquid on left, gas on right
        directionGL = twoPhaseEulerAdvection->fluxCalculatorLiquidGas->GetFluxCalculatorFunction()(
            twoPhaseEulerAdvection->fluxCalculatorLiquidGas->GetFluxCalculatorContext(), normalVelocityL, aL_L, densityL_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFluxGL, &p12GL);
    } else {
        // no discontinuous region
        massFluxGL = 0.0;
        p12GL = 0.5 * (p12GG + p12LL);
    }

    // Calculate total flux
    flux[CompressibleFlowFields::RHO] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * areaMag * (1 - alphaMin - alphaDif));

    PetscReal velMagL = MagVector(dim, velocityL);
    PetscReal velMagR = MagVector(dim, velocityR);
    // gas interface
    if (directionG == fluxCalculator::LEFT) {  // direction of GG,LL,LG should match since uniform velocity??
        PetscReal HG_L = internalEnergyG_L + velMagL * velMagL / 2.0 + p12GG / densityG_L;
        flux[CompressibleFlowFields::RHOE] = (HG_L * massFluxGG * areaMag * alphaMin);

        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] = velocityL[n] * areaMag * (massFluxGG * alphaMin) + (p12GG * alphaMin) * newnormal[n];
        }
    } else if (directionG == fluxCalculator::RIGHT) {
        PetscReal HG_R = internalEnergyG_R + velMagR * velMagR / 2.0 + p12GG / densityG_R;
        flux[CompressibleFlowFields::RHOE] = (HG_R * massFluxGG * areaMag * alphaMin);

        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] = velocityR[n] * areaMag * (massFluxGG * alphaMin) + (p12GG * alphaMin) * newnormal[n];
        }
    } else {
        PetscReal HG_L = internalEnergyG_L + velMagL * velMagL / 2.0 + p12GG / densityG_L;
        PetscReal HG_R = internalEnergyG_R + velMagR * velMagR / 2.0 + p12GG / densityG_R;

        flux[CompressibleFlowFields::RHOE] = (0.5 * (HG_L + HG_R) * massFluxGG * areaMag * alphaMin);
        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] = 0.5 * (velocityL[n] + velocityR[n]) * areaMag * (massFluxGG * alphaMin) + (p12GG * alphaMin) * newnormal[n];
        }
    }
    // add liquid interface
    if (directionL == fluxCalculator::LEFT) {  // direction of GG,LL,LG should match since uniform velocity???
        PetscReal HL_L = internalEnergyL_L + velMagL * velMagL / 2.0 + p12LL / densityL_L;
        flux[CompressibleFlowFields::RHOE] += (HL_L * massFluxLL * areaMag * alphaLiq);

        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] += velocityL[n] * areaMag * (massFluxLL * alphaLiq) + (p12LL * alphaLiq) * newnormal[n];
        }
    } else if (directionL == fluxCalculator::RIGHT) {
        PetscReal HL_R = internalEnergyL_R + velMagR * velMagR / 2.0 + p12LL / densityL_R;
        flux[CompressibleFlowFields::RHOE] += (HL_R * massFluxLL * areaMag * alphaLiq);

        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] += velocityR[n] * areaMag * (massFluxLL * alphaLiq) + (p12LL * alphaLiq) * newnormal[n];
        }
    } else {
        PetscReal HL_L = internalEnergyL_L + velMagL * velMagL / 2.0 + p12LL / densityL_L;
        PetscReal HL_R = internalEnergyL_R + velMagR * velMagR / 2.0 + p12LL / densityL_R;

        flux[CompressibleFlowFields::RHOE] += (0.5 * (HL_L + HL_R) * massFluxLL * areaMag * alphaLiq);
        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] += 0.5 * (velocityL[n] + velocityR[n]) * areaMag * (massFluxLL * alphaLiq) + (p12LL * alphaLiq) * newnormal[n];
        }
    }
    // add gas-liquid or liquid-gas interface
    if (directionGL == fluxCalculator::LEFT) {  // direction of GG,LL,LG should match since uniform velocity???
        PetscReal HGL_L;
        if (alphaL > alphaR) {
            // gas on left
            HGL_L = internalEnergyG_L + velMagL * velMagL / 2.0 + p12GL / densityG_L;
        } else if (alphaL < alphaR) {
            // liquid on left
            HGL_L = internalEnergyL_L + velMagL * velMagL / 2.0 + p12GL / densityL_L;
        } else {
            // no discontinuous region
            HGL_L = 0.0;
        }
        flux[CompressibleFlowFields::RHOE] += (HGL_L * massFluxGL * areaMag * alphaDif);

        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] += velocityL[n] * areaMag * (massFluxGL * alphaDif) + (p12GL * alphaDif) * newnormal[n];
        }
    } else if (directionGL == fluxCalculator::RIGHT) {
        PetscReal HGL_R;
        if (alphaL > alphaR) {
            // liquid on right
            HGL_R = internalEnergyL_R + velMagR * velMagR / 2.0 + p12GL / densityL_R;
        } else if (alphaL < alphaR) {
            // gas on right
            HGL_R = internalEnergyG_R + velMagR * velMagR / 2.0 + p12GL / densityG_R;
        } else {
            // no discontinuous region
            HGL_R = 0.0;
        }
        flux[CompressibleFlowFields::RHOE] += (HGL_R * massFluxGL * areaMag * alphaDif);

        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] += velocityR[n] * areaMag * (massFluxGL * alphaDif) + (p12GL * alphaDif) * newnormal[n];
        }
    } else {
        PetscReal HGL_L;
        if (alphaL > alphaR) {
            // gas on left
            HGL_L = internalEnergyG_L + velMagL * velMagL / 2.0 + p12GL / densityG_L;
        } else if (alphaL < alphaR) {
            // liquid on left
            HGL_L = internalEnergyL_L + velMagL * velMagL / 2.0 + p12GL / densityL_L;
        } else {
            // no discontinuous region
            HGL_L = 0.0;
        }
        PetscReal HGL_R;
        if (alphaL > alphaR) {
            // liquid on right
            HGL_R = internalEnergyL_R + velMagR * velMagR / 2.0 + p12GL / densityL_R;
        } else if (alphaL < alphaR) {
            // gas on right
            HGL_R = internalEnergyG_R + velMagR * velMagR / 2.0 + p12GL / densityG_R;
        } else {
            // no discontinuous region
            HGL_R = 0.0;
        }

        flux[CompressibleFlowFields::RHOE] += (0.5 * (HGL_L + HGL_R) * massFluxGL * areaMag * alphaDif);
        for (PetscInt n = 0; n < dim; n++) {
            flux[CompressibleFlowFields::RHOU + n] += 0.5 * (velocityL[n] + velocityR[n]) * areaMag * (massFluxGL * alphaDif) + (p12GL * alphaDif) * newnormal[n];
        }
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeVFFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscScalar *fieldL,
                                                                                                      const PetscScalar *fieldR, const PetscInt *aOff, const PetscScalar *auxL, const PetscScalar *auxR,
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
    PetscInt dim, const std::shared_ptr<eos::EOS> &eosGas, const std::shared_ptr<eos::EOS> &eosLiquid) {
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
    } else if (stiffenedGasEos1 && stiffenedGasEos2) {
        return std::make_shared<StiffenedGasStiffenedGasDecoder>(dim, stiffenedGasEos1, stiffenedGasEos2);
    }
    throw std::invalid_argument("Unknown combination of equation of states for ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseDecoder");
}

/**PerfectGasPerfectGasDecoder**************/
ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasPerfectGasDecoder::PerfectGasPerfectGasDecoder(PetscInt dim, const std::shared_ptr<eos::PerfectGas> &eosGas,
                                                                                                                  const std::shared_ptr<eos::PerfectGas> &eosLiquid)
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
ablate::finiteVolume::processes::TwoPhaseEulerAdvection::PerfectGasStiffenedGasDecoder::PerfectGasStiffenedGasDecoder(PetscInt dim, const std::shared_ptr<eos::PerfectGas> &eosGas,
                                                                                                                      const std::shared_ptr<eos::StiffenedGas> &eosLiquid)
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
//    if (Yg>1.0 || Yl>1.0){
//        throw std::invalid_argument("alpha is too big");
//    }

    PetscReal R1 = eosGas->GetGasConstant();
    PetscReal cp2 = eosLiquid->GetSpecificHeatCp();
    PetscReal p02 = eosLiquid->GetReferencePressure();
    PetscReal gamma1 = eosGas->GetSpecificHeatRatio();
    PetscReal gamma2 = eosLiquid->GetSpecificHeatRatio();
    PetscReal cv1 = R1 / (gamma1 - 1);
////    PetscReal cp1 = gamma1*cv1;
//    if (Yl<=1e-12){
//        PetscReal rhoG, rhoL, eG, eL;
//        rhoG = (*density);
//        eG = (*internalEnergy);
//        PetscReal TG = eG/cv1;
//        PetscReal pG = (gamma1-1.0)*rhoG*eG;
//        rhoL = gamma2/(gamma2-1)*(pG+p02)/cp2/TG;
//        eL = (pG+gamma2*p02)/(gamma2-1)/rhoL;
//        PetscReal a1 = PetscSqrtReal(gamma1*pG/rhoG);
//        PetscReal a2 = PetscSqrtReal(gamma2*(pG+p02)/rhoL);
//
//        // once state defined
//        *T=TG;
//        *densityG = rhoG;
//        *densityL = rhoL;
//        *internalEnergyG = eG;
//        *internalEnergyL = eL;
//        *alpha = densityVF / (*densityG);
//        *p = pG;  // pressure equilibrium, pG = pL
//        *aG = a1;
//        *aL = a2;
//        *MG = (*normalVelocity) / (*aG);
//        *ML = (*normalVelocity) / (*aL);
//    } else if ( Yl>=(1.0-1e-12) ){
//        PetscReal rhoG, rhoL, eG, eL;
//        rhoL = (*density);
//        eL = (*internalEnergy);
//        PetscReal TL = (eL-p02/rhoL)*gamma2/cp2;
//        PetscReal pL = (gamma2-1.0)*rhoL*eL-gamma2*p02;
//        rhoG = pL/TL/R1;
//        eG = cv1*TL;
//        PetscReal a1 = PetscSqrtReal(gamma1*pL/rhoG);
//        PetscReal a2 = PetscSqrtReal(gamma2*(pL+p02)/rhoL);
//
//        // once state defined
//        *T = TL;
//        *densityG = rhoG;
//        *densityL = rhoL;
//        *internalEnergyG = eG;
//        *internalEnergyL = eL;
//        *alpha = densityVF / (*densityG);
//        *p = pL;  // pressure equilibrium, pG = pL
//        *aG = a1;
//        *aL = a2;
//        *MG = (*normalVelocity) / (*aG);
//        *ML = (*normalVelocity) / (*aL);
//    }
//
//
//    else{
    PetscReal etot = (*internalEnergy);
    PetscReal A = cp2 / cv1 / gamma2;
    PetscReal B = Yg + Yl * A;
    PetscReal D = p02 / (*density) - etot;
    PetscReal E = Yg * p02 / (*density) + Yl * A * etot;
    PetscReal eG, eL, a, b, c, root1, root2;
    if (Yg < 0.5) {  // avoid divide by zero, 1E-3
        a = B * (Yg * (gamma2 - 1) - Yg * (gamma1 - 1) - gamma2 * B);
        b = etot * Yg * (gamma1 - 1) + etot * B + Yg * (gamma2 - 1) * D - gamma2 * D * B;
        c = etot * D;
        if (PetscAbs(a) < 1E-5) {
            eG = -c / b;
        } else {
            root1 = (-b + PetscSqrtReal(PetscSqr(b) - 4 * a * c)) / (2 * a);
            root2 = (-b - PetscSqrtReal(PetscSqr(b) - 4 * a * c)) / (2 * a);
            if (root1 > 1E-5 && root2 > 1E-5) {
                eG = PetscMax(root1, root2); // used to be Min
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
    PetscReal ar = (gamma1 - 1) * eG;
    PetscReal br = gamma2 * p02 - Yg*(*density)*eG*(gamma1 - 1) - Yl*(*density)*eL*(gamma2-1);
    PetscReal cr = -Yg*(*density)*gamma2*p02;
    PetscReal root1r = (-br + PetscSqrtReal(PetscSqr(br) - 4 * ar * cr)) / (2 * ar);
    PetscReal root2r = (-br - PetscSqrtReal(PetscSqr(br) - 4 * ar * cr)) / (2 * ar);
    PetscReal rhoG;
    if (PetscAbs(ar) < 1E-5) {
        rhoG = -cr / br;
    } else {
        rhoG = PetscMax(root1r, root2r);
        if (rhoG < 1E-5) {  // negative density not physical
            throw std::invalid_argument("ablate::finiteVolume::twoPhaseEulerAdvection PerfectGas/StiffenedGas DecodeState cannot result in negative density rhoG");
        }
    }
    PetscReal rhoL = ((gamma1 - 1) * eG * rhoG + gamma2 * p02) / (gamma2 - 1) / eL;
    if (rhoL < 1E-5) {  // negative density not physical
        throw std::invalid_argument("ablate::finiteVolume::twoPhaseEulerAdvection PerfectGas/StiffenedGas DecodeState cannot result in negative density rhoL");
    }
    PetscReal pG = 0;
    PetscReal pL;
    PetscReal a1 = 0;
    PetscReal a2 = 0;

//    if ( PetscAbs(Yg/rhoG+Yl/rhoL-1/(*density)) > 1e-10){
//        throw std::invalid_argument("rhoG and rhoL decoded do not conserve mass");
//    }
//    if ( PetscAbs(Yg*eG+Yl*eL-etot) > 1e-10) {
//        throw std::invalid_argument("eG and eL decoded do not conserve energy");
//    }
//        SNES snes;
//        Vec x, r;
//        Mat J;
//        VecCreate(PETSC_COMM_SELF, &x);
//        VecSetSizes(x, PETSC_DECIDE, 4);
//        VecSetFromOptions(x);
//        VecSet(x, (*internalEnergy)); // set initial guess to conserved density, [rho1, rho2, e1, e2] = [rho, rho, rho, rho]
//        VecSetValue(x, 0, 1.0, INSERT_VALUES);
//        VecSetValue(x, 1, 1000.0, INSERT_VALUES);
//        VecDuplicate(x, &r);
//
//        MatCreate(PETSC_COMM_SELF, &J);
//        MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 4, 4);
//        MatSetFromOptions(J);
//        MatSetUp(J);
//
//        SNESCreate(PETSC_COMM_SELF, &snes);
//        DecodeDataStructGas decodeDataStruct{.etot = (*internalEnergy),
//            .rhotot = (*density),
//            .Yg = densityVF / (*density),
//            .Yl = ((*density) - densityVF) / (*density),
//            .gam1 = gamma1,
//            .gam2 = gamma2,
//            .cvg = cv1,
//            .cpl = cp2,
//            .p0l = p02,
//        };
//        SNESSetFunction(snes, r, FormFunctionGas, &decodeDataStruct);
//        SNESSetJacobian(snes, J, J, FormJacobianGas, &decodeDataStruct);
//        // default Newton's method
//        //      SNESSetType(snes, "newtontr");
//        //      SNESSetTolerances(SNES snes, PetscReal atol, PetscReal rtol, PetscReal stol, PetscInt its, PetscInt fcts);
//        //          default rtol = 10e-8
//        // snes_fd: use FD Jacobian - SNESComputeJacobianDefault()
//        // snes_monitor : view residuals for each iteration
//    //    PetscOptionsSetValue(NULL, "-snes_monitor", NULL);
//    //    PetscOptionsSetValue(NULL, "-snes_converged_reason", NULL);
//        SNESSetFromOptions(snes);
//        SNESSolve(snes, NULL, x);
//    //    VecView(x, PETSC_VIEWER_STDOUT_SELF); // output solution
//        const PetscScalar *ax;
//        VecGetArrayRead(x, &ax);
//        PetscReal rhoG = ax[0];
//        PetscReal rhoL = ax[1];
//        PetscReal eG = ax[2];
//        PetscReal eL = ax[3];
//
//        SNESDestroy(&snes);
//        VecDestroy(&x);
//        VecDestroy(&r);
//        MatDestroy(&J);
//
//        PetscReal etG = eG + ke;
//        PetscReal etL = eL + ke;
//
//        PetscReal pG = 0;
//        PetscReal pL;
//        PetscReal a1 = 0;
//        PetscReal a2 = 0;

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
//    if (*alpha > 1.0+1E-8){
//        *alpha = 1.0;
//        throw std::invalid_argument("alpha calculated is greater than 1");
//    }
//    if (PetscAbs((*density)-(*alpha)*rhoG-(1-(*alpha))*rhoL)>1e-10){
//        throw std::invalid_argument("volume fraction not conserved");
//    }
    *p = pG;  // pressure equilibrium, pG = pL
//    if (PetscAbs(pG-pL)>1e-10){
//        throw std::invalid_argument("pressure equilibrium not achieved");
//    }
    *aG = a1;
    *aL = a2;
    *MG = (*normalVelocity) / (*aG);
    *ML = (*normalVelocity) / (*aL);
//}
    PetscReal a1t, a2t, ainv, amix, bmodt;
    a1t = PetscSqrtReal((gamma1-1)*cv1*(*T));
    a2t = PetscSqrtReal((gamma2-1)/gamma2*cp2*(*T));
    ainv = Yg/(rhoG*rhoG*a1t*a1t) + Yl/(rhoL*rhoL*a2t*a2t);
    amix = PetscSqrtReal(1/ainv)/(*density);
    bmodt = (*density)*amix*amix;
    if (bmodt < 0.0){
        throw std::invalid_argument("isothermal bulk modulus of mixture negative");
    }

}

/**StiffenedGasStiffenedGasDecoder**************/
ablate::finiteVolume::processes::TwoPhaseEulerAdvection::StiffenedGasStiffenedGasDecoder::StiffenedGasStiffenedGasDecoder(PetscInt dim, const std::shared_ptr<eos::StiffenedGas> &eosGas,
                                                                                                                      const std::shared_ptr<eos::StiffenedGas> &eosLiquid)
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
void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::StiffenedGasStiffenedGasDecoder::DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues,
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

//    // mass fractions
//    PetscReal Yg = densityVF / (*density);
//    PetscReal Yl = ((*density) - densityVF) / (*density);

    PetscReal cp1 = eosGas->GetSpecificHeatCp();
    PetscReal cp2 = eosLiquid->GetSpecificHeatCp();
    PetscReal p01 = eosGas->GetReferencePressure();
    PetscReal p02 = eosLiquid->GetReferencePressure();
    PetscReal gamma1 = eosGas->GetSpecificHeatRatio();
    PetscReal gamma2 = eosLiquid->GetSpecificHeatRatio();

    SNES snes;
    Vec x, r;
    Mat J;
    VecCreate(PETSC_COMM_SELF, &x);
    VecSetSizes(x, PETSC_DECIDE, 4);
    VecSetFromOptions(x);
    VecSet(x, (*density)); // set initial guess to conserved density, [rho1, rho2, e1, e2] = [rho, rho, rho, rho]
    VecDuplicate(x, &r);

    MatCreate(PETSC_COMM_SELF, &J);
    MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 4, 4);
    MatSetFromOptions(J);
    MatSetUp(J);

    SNESCreate(PETSC_COMM_SELF, &snes);
    DecodeDataStructStiff decodeDataStruct{.etot = (*internalEnergy),
        .rhotot = (*density),
        .Yg = densityVF / (*density),
        .Yl = ((*density) - densityVF) / (*density),
        .gam1 = gamma1,
        .gam2 = gamma2,
        .cpg = cp1,
        .cpl = cp2,
        .p0g = p01,
        .p0l = p02,
    };
    SNESSetFunction(snes, r, FormFunctionStiff, &decodeDataStruct);
    SNESSetJacobian(snes, J, J, FormJacobianStiff, &decodeDataStruct);
    // default Newton's method
    //      SNESSetType(snes, "newtontr");
    //      SNESSetTolerances(SNES snes, PetscReal atol, PetscReal rtol, PetscReal stol, PetscInt its, PetscInt fcts);
    //          default rtol = 10e-8
    // snes_fd: use FD Jacobian - SNESComputeJacobianDefault()
    // snes_monitor : view residuals for each iteration
//    PetscOptionsSetValue(NULL, "-snes_monitor", NULL);
//    PetscOptionsSetValue(NULL, "-snes_converged_reason", NULL);
    SNESSetFromOptions(snes);
    SNESSolve(snes, NULL, x);
//    VecView(x, PETSC_VIEWER_STDOUT_SELF); // output solution
    const PetscScalar *ax;
    VecGetArrayRead(x, &ax);
    PetscReal rhoG = ax[0];
    PetscReal rhoL = ax[1];
    PetscReal eG = ax[2];
    PetscReal eL = ax[3];

    SNESDestroy(&snes);
    VecDestroy(&x);
    VecDestroy(&r);
    MatDestroy(&J);

    PetscReal etG = eG + ke;
    PetscReal etL = eL + ke;

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

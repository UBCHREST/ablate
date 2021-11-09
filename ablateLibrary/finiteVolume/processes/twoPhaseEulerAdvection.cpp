#include "twoPhaseEulerAdvection.hpp"
// #include <petscsnes.h>

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

struct DecodeDataStruct {
    std::shared_ptr<ablate::eos::EOS> eosGas;
    std::shared_ptr<ablate::eos::EOS> eosLiquid;
    PetscReal ke;
    PetscReal e;
    PetscReal rho;
    PetscReal Yg;
    PetscReal Yl;
    PetscInt dim;
    PetscReal *vel;
};

PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    auto decodeDataStruct = (DecodeDataStruct *)ctx;
    const PetscReal *ax;
    PetscReal *aF;
    VecGetArrayRead(x, &ax);
    // ax = [rhog, eg]
    PetscReal rhoG = ax[0];
    PetscReal eG = ax[1];
    PetscReal etG = ax[1] + decodeDataStruct->ke;
    PetscReal eL = (decodeDataStruct->e - decodeDataStruct->Yg * ax[1]) / (decodeDataStruct->Yl + 1E-10);
    PetscReal etL = eL + decodeDataStruct->ke;
    PetscReal rhoL = decodeDataStruct->Yl / (1 / decodeDataStruct->rho - decodeDataStruct->Yg / ax[0] + 1E-10) + 1E-10;

    PetscReal massfluxG[3];
    PetscReal massfluxL[3];
    for (PetscInt d = 0; d < decodeDataStruct->dim; d++) {
        massfluxG[d] = decodeDataStruct->vel[d] * rhoG;
        massfluxL[d] = decodeDataStruct->vel[d] * rhoL;
    }

    PetscReal pG;
    PetscReal pL;
    PetscReal TG;
    PetscReal TL;
    PetscReal aG;
    PetscReal aL;

    decodeDataStruct->eosGas->GetDecodeStateFunction()(decodeDataStruct->dim, rhoG, etG, decodeDataStruct->vel, NULL, &eG, &aG, &pG, decodeDataStruct->eosGas->GetDecodeStateContext());
    decodeDataStruct->eosGas->GetComputeTemperatureFunction()(decodeDataStruct->dim, rhoG, etG, massfluxG, NULL, &TG, decodeDataStruct->eosGas->GetComputeTemperatureContext());
    decodeDataStruct->eosLiquid->GetDecodeStateFunction()(decodeDataStruct->dim, rhoL, etL, decodeDataStruct->vel, NULL, &eL, &aL, &pL, decodeDataStruct->eosLiquid->GetDecodeStateContext());
    decodeDataStruct->eosLiquid->GetComputeTemperatureFunction()(decodeDataStruct->dim, rhoL, etL, massfluxL, NULL, &TL, decodeDataStruct->eosLiquid->GetComputeTemperatureContext());

    VecGetArray(F, &aF);
    aF[0] = pG - pL;
    aF[1] = TG - TL;
    VecRestoreArrayRead(x, &ax);
    VecRestoreArray(F, &aF);
    return 0;
}

void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DecodeTwoPhaseEulerState(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid, PetscInt dim,
                                                                                       const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, PetscReal *densityG,
                                                                                       PetscReal *densityL, PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy,
                                                                                       PetscReal *internalEnergyG, PetscReal *internalEnergyL, PetscReal *aG, PetscReal *aL, PetscReal *MG,
                                                                                       PetscReal *ML, PetscReal *p, PetscReal *alpha) {
    const int EULER_FIELD = 1;  // denstiyVF is [0] field
    // (densityVF, RHO, RHOE, RHOU, RHOV, RHOW)
    // decode
    PetscReal densityVF = conservedValues[0];
    *density = conservedValues[0 + EULER_FIELD];
    PetscReal totalEnergy = conservedValues[1 + EULER_FIELD] / (*density);

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
    // mass fractions
    PetscReal Yg = densityVF / (*density);
    PetscReal Yl = ((*density) - densityVF) / (*density);

    // additional equations:
    // 1/density = Yg/densityG + Yl/densityL;
    // internalEnergy = Yg*internalEnergyG + Yl*internalEnergyL;

    SNES snes;  // nonlinear solver
    Vec x, r;   // solution, residual vectors
    VecCreate(PETSC_COMM_SELF, &x);
    VecSetSizes(x, PETSC_DECIDE, 2);  // 2x1 vector
    VecSetFromOptions(x);
    VecSet(x, 1.0);  // set initial guess [rho1, e1]= [1.0,1.0]
                     //    VecSetValues(x, 2, [0,1],[1.0,1.0],INSERT_VALUES); // set each initial guess separately
    VecDuplicate(x, &r);

    SNESCreate(PETSC_COMM_SELF, &snes);
    DecodeDataStruct decodeDataStruct{.eosGas = eosGas,
                                      .eosLiquid = eosLiquid,
                                      .ke = ke,
                                      .e = (*internalEnergy),
                                      .rho = (*density),
                                      .Yg = densityVF / (*density),  // mass fractions
                                      .Yl = ((*density) - densityVF) / (*density),
                                      .dim = dim,
                                      .vel = velocity};
    SNESSetFunction(snes, r, FormFunction, &decodeDataStruct);
    // default Newton's method, SNESSetType(SNES snes, SNESType method);
//    SNESSetType(snes,"newtontr");
    //    SNESSetTolerances(SNES snes,PetscReal atol,PetscReal rtol,PetscReal stol, PetscInt its,PetscInt fcts);
    SNESSetTolerances(snes, 1E-16, 1E-26, 1E-16, 100000, 100000);
    // default rtol=10e-8
    // snes_fd : use FD Jacobian - SNESComputeJacobianDefault()
    // snes_monitor : view residuals for each iteration
    PetscOptionsSetValue(NULL, "-snes_monitor", NULL);
    PetscOptionsSetValue(NULL, "-snes_converged_reason", NULL);
    SNESSetFromOptions(snes);
    //    SNESMonitorSet(SNES snes,PetscErrorCode (*mon)(SNES,PetscInt its,PetscReal norm,void* mctx),void *mctx,PetscErrorCode (*monitordestroy)(void**));

    SNESSolve(snes, NULL, x);              // getting 1.93116, 928993
                                           // fixed mf, 1.8735, 938779
                                           // want 1.88229965, 937273
    VecView(x, PETSC_VIEWER_STDOUT_SELF);  // output solution
    const PetscScalar *ax;
    VecGetArrayRead(x, &ax);

    // guess et1, rho1; calculate p, T from EOS and
    // set of equations
    //      pG-pl = 0
    //      Tg-Tl = 0

    //  ax = [rhog, eg]
    PetscReal eG = ax[1];
    PetscReal etG = eG + ke;
    PetscReal eL = ((*internalEnergy) - Yg * eG) / (Yl + 1E-10);
    PetscReal etL = eL + ke;
    PetscReal rhoG = ax[0];
    PetscReal rhoL = Yl / (1 / (*density) - Yg / rhoG + 1E-10) + 1E-10;
    // define output variables
    PetscReal pG;
    PetscReal pL;
    PetscReal a1;
    PetscReal a2;
    eosGas->GetDecodeStateFunction()(dim, rhoG, etG, velocity, NULL, &eG, &a1, &pG, eosGas->GetDecodeStateContext());
    eosLiquid->GetDecodeStateFunction()(dim, rhoL, etL, velocity, NULL, &eL, &a2, &pL, eosLiquid->GetDecodeStateContext());

    SNESDestroy(&snes);
    VecDestroy(&x);
    VecDestroy(&r);

    // once state defined
    *densityG = rhoG;
    *densityL = rhoL;
    *internalEnergyG = eG;
    *internalEnergyL = eL;
    *p = (pG + pL) / 2;
    *aG = a1;
    *aL = a2;
    *MG = (*normalVelocity) / (*aG);
    *ML = (*normalVelocity) / (*aL);
    *alpha = densityVF / (*densityG);
}

ablate::finiteVolume::processes::TwoPhaseEulerAdvection::TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid,
                                                                                std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid)
    : eosGas(eosGas), eosLiquid(eosLiquid), fluxCalculatorGasGas(fluxCalculatorGasGas), fluxCalculatorGasLiquid(fluxCalculatorGasLiquid), fluxCalculatorLiquidLiquid(fluxCalculatorLiquidLiquid) {}
void ablate::finiteVolume::processes::TwoPhaseEulerAdvection::Initialize(ablate::finiteVolume::FiniteVolume &flow) {
    // Currently, no option for species advection
    flow.RegisterRHSFunction(CompressibleFlowComputeEulerFlux, this, "euler", {"densityVF", "euler"}, {});
    flow.RegisterRHSFunction(CompressibleFlowComputeVFFlux, this, "densityVF", {"densityVF", "euler"}, {});
}
PetscErrorCode ablate::finiteVolume::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x,
                                                                                                         const PetscScalar *fieldL, const PetscScalar *fieldR, const PetscScalar *gradL,
                                                                                                         const PetscScalar *gradR, const PetscInt *aOff, const PetscInt *aOff_x,
                                                                                                         const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                                         const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
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
    DecodeTwoPhaseEulerState(twoPhaseEulerAdvection->eosGas,
                             twoPhaseEulerAdvection->eosLiquid,
                             dim,
                             fieldL + uOff[EULER_FIELD],
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
    PetscReal alphaR;
    DecodeTwoPhaseEulerState(twoPhaseEulerAdvection->eosGas,
                             twoPhaseEulerAdvection->eosLiquid,
                             dim,
                             fieldR + uOff[EULER_FIELD],
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
                             &alphaR);

    // get the face values
    PetscReal massFluxGG;
    PetscReal massFluxGL;
    PetscReal massFluxLL;
    PetscReal p12;  // pressure equilibrium ( might need to make sure they match)

    // call flux calculator 3 times, gas-gas, gas-liquid, liquid-liquid regions
    fluxCalculator::Direction directionG = twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFluxGG, &p12);
    //    fluxCalculator::Direction directionL =  // should be same direction, if not, big problem
    twoPhaseEulerAdvection->fluxCalculatorLiquidLiquid->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorLiquidLiquid->GetFluxCalculatorContext(), normalVelocityL, aL_L, densityL_L, pL, normalVelocityR, aL_R, densityL_R, pR, &massFluxLL, &p12);
    if (alphaL > alphaR) {
        // gas on left, liquid on right
        twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorFunction()(
            twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aL_R, densityL_R, pR, &massFluxGL, &p12);
    } else if (alphaL < alphaR) {
        // liquid on left, gas on right
        twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorFunction()(
            twoPhaseEulerAdvection->fluxCalculatorGasLiquid->GetFluxCalculatorContext(), normalVelocityL, aL_L, densityL_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFluxGL, &p12);
    } else {
        // no discontinuous region
        massFluxGL = 0.0;
    }

    // Calculate total flux
    PetscReal alphaMin = PetscMin(alphaR, alphaL);
    PetscReal alphaDif = PetscAbs(alphaL - alphaR);
    if (directionG == fluxCalculator::LEFT) {  // direction of GG,LL,LG should match since uniform velocity???
                                               //        flux[RHO] = massFlux * areaMag;
        flux[0 + EULER_FIELD] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * (1 - alphaMin));
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
        flux[1 + EULER_FIELD] = (HG_L * massFluxGG * areaMag * alphaMin) + (HGL_L * massFluxGL * areaMag * alphaDif) + (HL_L * massFluxLL * areaMag * (1 - alphaMin));
        for (PetscInt n = 0; n < dim; n++) {
            //            flux[RHOU + n] = velocityL[n] * massFlux * areaMag + p12 * fg->normal[n];
            flux[2 + EULER_FIELD + n] = velocityL[n] * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1 - alphaMin)) + p12 * fg->normal[n];
        }
    } else if (directionG == fluxCalculator::RIGHT) {
        //        flux[RHO] = massFlux * areaMag;
        flux[0 + EULER_FIELD] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * (1 - alphaMin));
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
        flux[1 + EULER_FIELD] = (HG_R * massFluxGG * areaMag * alphaMin) + (HGL_R * massFluxGL * areaMag * alphaDif) + (HL_R * massFluxLL * areaMag * (1 - alphaMin));
        for (PetscInt n = 0; n < dim; n++) {
            //            flux[RHOU + n] = velocityR[n] * massFlux * areaMag + p12 * fg->normal[n];
            flux[2 + EULER_FIELD + n] = velocityR[n] * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1 - alphaMin)) + p12 * fg->normal[n];
        }
    } else {
        //        flux[RHO] = massFlux * areaMag;
        flux[0 + EULER_FIELD] = (massFluxGG * areaMag * alphaMin) + (massFluxGL * areaMag * alphaDif) + (massFluxLL * (1 - alphaMin));

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
        flux[1 + EULER_FIELD] =
            (0.5 * (HG_L + HG_R) * massFluxGG * areaMag * alphaMin) + (0.5 * (HGL_L + HGL_R) * massFluxGL * areaMag * alphaDif) + (0.5 * (HL_L + HL_R) * massFluxLL * areaMag * (1 - alphaMin));

        for (PetscInt n = 0; n < dim; n++) {
            //            flux[RHOU + n] = 0.5 * (velocityL[n] + velocityR[n]) * massFlux * areaMag + p12 * fg->normal[n];
            flux[2 + EULER_FIELD + n] = 0.5 * (velocityL[n] + velocityR[n]) * areaMag * (massFluxGG * alphaMin + massFluxGL * alphaDif + massFluxLL * (1 - alphaMin)) + p12 * fg->normal[n];
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
    //    const int VF_FIELD = 0;
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
    DecodeTwoPhaseEulerState(twoPhaseEulerAdvection->eosGas,
                             twoPhaseEulerAdvection->eosLiquid,
                             dim,
                             fieldR + uOff[EULER_FIELD],
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
    PetscReal alphaR;
    DecodeTwoPhaseEulerState(twoPhaseEulerAdvection->eosGas,
                             twoPhaseEulerAdvection->eosLiquid,
                             dim,
                             fieldR + uOff[EULER_FIELD],
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
                             &alphaR);

    // get the face values
    PetscReal massFlux;
    PetscReal p12;
    // calculate gas sub-area of face (stratified flow model)
    PetscReal alpha = PetscMin(alphaR, alphaL);

    // how do I check that densityVF/dt is being integrated over Volume*alpha????? (can I get this to cancel out??)
    twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorFunction()(
        twoPhaseEulerAdvection->fluxCalculatorGasGas->GetFluxCalculatorContext(), normalVelocityL, aG_L, densityG_L, pL, normalVelocityR, aG_R, densityG_R, pR, &massFlux, &p12);
    flux[0] = massFlux * areaMag * alpha;  // is this correct???

    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::TwoPhaseEulerAdvection, "", ARG(ablate::eos::EOS, "eosGas", ""), ARG(ablate::eos::EOS, "eosLiquid", ""),
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorGasGas", ""), ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorGasLiquid", ""),
         ARG(ablate::finiteVolume::fluxCalculator::FluxCalculator, "fluxCalculatorLiquidLiquid", ""));

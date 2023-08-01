#include "riemannStiff.hpp"
#include <eos/perfectGas.hpp>
#include <eos/stiffenedGas.hpp>
#include "riemannCommon.hpp"
#include <signal.h>

ablate::finiteVolume::fluxCalculator::Direction ablate::finiteVolume::fluxCalculator::RiemannStiff::RiemannStiffFluxFunction(void *ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                                                                                                             PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                                                                                                             PetscReal *massFlux,

                                                                                                                             PetscReal *p12) {
    /*
     * gammaL: specific heat ratio for gas on left (pass in from EOS)
     * gamLm1 = gammaL - 1
     * gamLp1 = gammaL + 1
     * gammaR: specific heat ratio for stiffened gas on right (pass in from EOS)
     * gamRm1 = gammaR - 1
     * gamRp1 = gammaR + 1
     * p0L: reference pressure for stiffened gas on left (pass in from EOS)
     * p0R: reference pressure for stiffened gas on right (pass in from EOS)
     * uL: velocity on the left cell center
     * uR: velocity on the right cell center
     * rhoR: density on the right cell center
     * rhoL: density on the left cell center
     * pR: pressure on the right cell center
     * pL: pressure on the left cell center
     * aR: SoS on the right center cell
     * aL: SoS on the left center cell
     * pstar: pressure across contact surface
     * ustar: velocity across contact surface
     * rhostarR: density on the right of the contact surface
     * whostarL: density on the left of the contact surface
     * err: final residual for iteration
     * MAXIT: maximum iteration times
     */


    PetscInt i = 0;
    const PetscInt MAXIT = 100;
    const PetscReal err = 1e-6;
    auto gammaVec = (PetscReal *)ctx;  // pass-in specific heat ratio from EOS_Left
    // This is where Riemann solver lives.
    PetscReal gammaL = gammaVec[0];
    PetscReal gamLm1 = gammaL - 1.0, gamLp1 = gammaL + 1.0;
    PetscReal gammaR = gammaVec[1];
    PetscReal gamRm1 = gammaR - 1.0, gamRp1 = gammaR + 1.0;
    PetscReal p0L = gammaVec[2];
    PetscReal p0R = gammaVec[3];

    PetscReal pold, pstar, f_L_0, f_L_1, f_R_0, f_R_1, del_u = uR - uL;

    // Here is the initial guess for pstar - average of left and right pressures
    pstar = 0.5 * (pR + pL);

    ExpansionShockCalculation(pstar, gammaL, gamLm1, gamLp1, p0L, pL, aL, rhoL, &f_L_0, &f_L_1);
    ExpansionShockCalculation(pstar, gammaR, gamRm1, gamRp1, p0R, pR, aR, rhoR, &f_R_0, &f_R_1);

    // iteration starts
    while (PetscAbsReal(f_L_0 + f_R_0 + del_u) > err && i <= MAXIT)  // Newton's method
    {
        pold = pstar;
        pstar = pold - 0.5*(f_L_0 + f_R_0 + del_u) / (f_L_1 + f_R_1);  // new guess

        if (pstar < 0) {  // correct if negative pstar
            if (p0L > 0 && p0R > 0) {
                pstar = pold - (f_L_0 + f_R_0 + del_u) / (f_L_1 + f_R_1);
            } else {
                pstar = err;
            }
        }

        ExpansionShockCalculation(pstar, gammaL, gamLm1, gamLp1, p0L, pL, aL, rhoL, &f_L_0, &f_L_1);
        ExpansionShockCalculation(pstar, gammaR, gamRm1, gamRp1, p0R, pR, aR, rhoR, &f_R_0, &f_R_1);

        i++;
    }


    if (i > MAXIT) {

printf("uL: %+f\n", uL);
printf("aL: %+f\n", aL);
printf("rhoL: %+f\n", rhoL);
printf("pL: %+f\n", pL);
printf("uR: %+f\n", uR);
printf("aR: %+f\n", aR);
printf("rhoR: %+f\n", rhoR);
printf("pR: %+f\n", pR);
raise(SIGSEGV);
PetscFPrintf(MPI_COMM_WORLD, stderr, "(%s:%d, %s)\n", __FILE__, __LINE__, __FUNCTION__);
exit(0);
        throw std::runtime_error("Can't find pstar; Iteration not converging; Go back and do it again");
    }

    return riemannDirection(pstar, uL, aL, rhoL, p0L, pL, gammaL, f_L_0, uR, aR, rhoR, p0R, pR, gammaR, f_R_0, massFlux, p12);
}
ablate::finiteVolume::fluxCalculator::RiemannStiff::RiemannStiff(std::shared_ptr<eos::EOS> eosL, std::shared_ptr<eos::EOS> eosR) {
    auto perfectGasEosL = std::dynamic_pointer_cast<eos::PerfectGas>(eosL);
    auto perfectGasEosR = std::dynamic_pointer_cast<eos::PerfectGas>(eosR);
    auto stiffenedGasEosL = std::dynamic_pointer_cast<eos::StiffenedGas>(eosL);
    auto stiffenedGasEosR = std::dynamic_pointer_cast<eos::StiffenedGas>(eosR);
    if (!perfectGasEosL && !stiffenedGasEosL) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::RiemannStiff left only accepts EOS of type eos::PerfectGas or eos::StiffenedGas");
    }
    if (!perfectGasEosR && !stiffenedGasEosR) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::RiemannStiff right only accepts EOS of type eos::PerfectGas or eos::StiffenedGas");
    }
    if (perfectGasEosL) {
        gammaVec[0] = perfectGasEosL->GetSpecificHeatRatio();
        gammaVec[2] = 0;
    } else if (stiffenedGasEosL) {
        gammaVec[0] = stiffenedGasEosL->GetSpecificHeatRatio();
        gammaVec[2] = stiffenedGasEosL->GetReferencePressure();
    }
    if (perfectGasEosR) {
        gammaVec[1] = perfectGasEosR->GetSpecificHeatRatio();
        gammaVec[3] = 0;
    } else if (stiffenedGasEosR) {
        gammaVec[1] = stiffenedGasEosR->GetSpecificHeatRatio();
        gammaVec[3] = stiffenedGasEosR->GetReferencePressure();
    }
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::fluxCalculator::FluxCalculator, ablate::finiteVolume::fluxCalculator::RiemannStiff, "Exact Riemann Solution for 2 Stiffened Gasses",
         ARG(ablate::eos::EOS, "eosL", "only valid for perfect or stiffened gas"), ARG(ablate::eos::EOS, "eosR", "only valid for perfect or stiffened gas"));

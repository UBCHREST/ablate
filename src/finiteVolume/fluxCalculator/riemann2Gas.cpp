#include "riemann2Gas.hpp"
#include <eos/perfectGas.hpp>
#include "riemannCommon.hpp"

ablate::finiteVolume::fluxCalculator::Direction ablate::finiteVolume::fluxCalculator::Riemann2Gas::Riemann2GasFluxFunction(void *ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                                                                                                           PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                                                                                                           PetscReal *massFlux,

                                                                                                                           PetscReal *p12) {
    /*
     * gammaL: specific heat ratio for gas on left (pass in from EOS)
     * gamLm1 = gammaL - 1
     * gamLp1 = gammaL + 1
     * gammaR: specific heat ratio for gas on right (pass in from EOS)
     * gamRm1 = gammaR - 1
     * gamRp1 = gammaR + 1
     * uL: velocity on the left cell center
     * uR: velocity on the right cell center
     * rhoR: density on the right cell center
     * rhoL: density on the left cell center
     * pR: pressure on the right cell center
     * pL: pressure on the left cell center
     * aR: SoS on the  right center cell
     * aL: SoS on the left center cell
     * pstar: pressure across contact surface
     * ustar: velocity across contact surface
     * rhostarR: density on the right of the contact surface
     * whostarL: density on the left of the contact surface
     * err: final residual for iteration
     * MAXIT: maximum iteration times
     */

    auto gammaVec = (PetscReal *)ctx;  // pass-in specific heat ratio from EOS_Left
    // This is where Riemann solver lives.
    PetscReal gammaL = gammaVec[0];
    PetscReal gammaR = gammaVec[1];
    PetscReal pstar;

    // Here is the initial guess for pstar - assuming two exapansion wave (need to change for 2 gasses)
    //    pstar = aL + aR - (0.5 * gamLm1 * (uR - uL));
    //    pstar = pstar / ((aL / PetscPowReal(pL, 0.5 * gamLm1 / gammaL)) + (aR / PetscPowReal(pR, 0.5 * gamRm1 / gammaR)));
    //    pstar = PetscPowReal(pstar, 2.0 * gammaL / gamLm1);
    pstar = 0.5 * (pR + pL);

    return reimannSolver(uL, aL, rhoL, 0, pL, gammaL, uR, aR, rhoR, 0, pR, gammaR, pstar, massFlux, p12);
}
ablate::finiteVolume::fluxCalculator::Riemann2Gas::Riemann2Gas(std::shared_ptr<eos::EOS> eosL, std::shared_ptr<eos::EOS> eosR) {
    auto perfectGasEosL = std::dynamic_pointer_cast<eos::PerfectGas>(eosL);
    auto perfectGasEosR = std::dynamic_pointer_cast<eos::PerfectGas>(eosR);
    if (!perfectGasEosL) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::Riemann2Gas left only accepts EOS of type eos::PerfectGas");
    }
    if (!perfectGasEosR) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::Riemann2Gas right only accepts EOS of type eos::PerfectGas");
    }
    gammaVec[0] = perfectGasEosL->GetSpecificHeatRatio();
    gammaVec[1] = perfectGasEosR->GetSpecificHeatRatio();
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::fluxCalculator::FluxCalculator, ablate::finiteVolume::fluxCalculator::Riemann2Gas, "Exact Riemann Solution for 2 Perfect Gasses",
         ARG(ablate::eos::EOS, "eosL", "only valid for perfect gas"), ARG(ablate::eos::EOS, "eosR", "only valid for perfect gas"));

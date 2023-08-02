#include "riemannStiff.hpp"
#include <signal.h>
#include <eos/perfectGas.hpp>
#include <eos/stiffenedGas.hpp>
#include "riemannCommon.hpp"

ablate::finiteVolume::fluxCalculator::Direction ablate::finiteVolume::fluxCalculator::RiemannStiff::RiemannStiffFluxFunction(void *ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                                                                                                             PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                                                                                                             PetscReal *massFlux,

                                                                                                                             PetscReal *p12) {
    /*
     * gammaL: specific heat ratio for gas on left (pass in from EOS)
     * gammaR: specific heat ratio for stiffened gas on right (pass in from EOS)
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
     */

    auto gammaVec = (PetscReal *)ctx;  // pass-in specific heat ratio from EOS_Left
    // This is where Riemann solver lives.
    PetscReal gammaL = gammaVec[0];
    PetscReal gammaR = gammaVec[1];
    PetscReal p0L = gammaVec[2];
    PetscReal p0R = gammaVec[3];

    // Here is the initial guess for pstar - average of left and right pressures
    PetscReal pstar = 0.5 * (pR + pL);

    return riemannSolver(uL, aL, rhoL, p0L, pL, gammaL, uR, aR, rhoR, p0R, pR, gammaR, pstar, massFlux, p12);
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

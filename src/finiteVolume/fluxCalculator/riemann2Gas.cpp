#include "riemann2Gas.hpp"
#include <eos/perfectGas.hpp>

using namespace ablate::finiteVolume::fluxCalculator;

Direction Riemann2Gas::Riemann2GasFluxFunction(void *ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux,

                                               PetscReal *p12) {
    /*
     * gammaL: specific heat ratio for gas on left (pass in from EOS)
     * gammaR: specific heat ratio for gas on right (pass in from EOS)
     * uL: velocity on the left cell center
     * uR: velocity on the right cell center
     * rhoR: density on the right cell center
     * rhoL: density on the left cell center
     * pR: pressure on the right cell center
     * pL: pressure on the left cell center
     * aR: SoS on the  right center cell
     * aL: SoS on the left center cell
     * pstar: pressure across contact surface
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

    Direction dir = riemannSolver(uL, aL, rhoL, 0, pL, gammaL, uR, aR, rhoR, 0, pR, gammaR, pstar, massFlux, p12);

    return dir;
}
Riemann2Gas::Riemann2Gas(std::shared_ptr<eos::EOS> eosL, std::shared_ptr<eos::EOS> eosR) {
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
REGISTER(FluxCalculator, Riemann2Gas, "Exact Riemann Solution for 2 Perfect Gasses", ARG(ablate::eos::EOS, "eosL", "only valid for perfect gas"),
         ARG(ablate::eos::EOS, "eosR", "only valid for perfect gas"));

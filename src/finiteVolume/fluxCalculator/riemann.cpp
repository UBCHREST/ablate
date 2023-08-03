#include "riemann.hpp"
#include <eos/perfectGas.hpp>

using namespace ablate::finiteVolume::fluxCalculator;


Direction Riemann::RiemannFluxFunction(void *ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux, PetscReal *p12) {
    /*
     * gamma: specific heat ratio (pass in from EOS)
     * gamm1 = gamma - 1
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
     */

    PetscReal gamma = *(PetscReal *)ctx;  // pass-in specific heat ratio from EOS
    // This is where Rieman solver lives.
    PetscReal gamm1 = gamma - 1.0;
    PetscReal pstar;

    // Here is the initial guess for pstar - assuming two exapansion wave
    pstar = aL + aR - (0.5 * gamm1 * (uR - uL));
    pstar = pstar / ((aL / PetscPowReal(pL, 0.5 * gamm1 / gamma)) + (aR / PetscPowReal(pR, 0.5 * gamm1 / gamma)));
    pstar = PetscPowReal(pstar, 2.0 * gamma / gamm1);

    Direction dir = riemannSolver(uL, aL, rhoL, 0, pL, gamma, uR, aR, rhoR, 0, pR, gamma, pstar, massFlux, p12);

    return dir;

}
Riemann::Riemann(std::shared_ptr<eos::EOS> eosIn) {
    auto perfectGasEos = std::dynamic_pointer_cast<eos::PerfectGas>(eosIn);
    if (!perfectGasEos) {
        throw std::invalid_argument("ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::Rieman only accepts EOS of type eos::PerfectGas");
    }
    gamma = perfectGasEos->GetSpecificHeatRatio();
}

#include "registrar.hpp"
REGISTER(FluxCalculator, Riemann, "Exact Riemann Solution", ARG(ablate::eos::EOS, "eos", "only valid for perfect gas"));

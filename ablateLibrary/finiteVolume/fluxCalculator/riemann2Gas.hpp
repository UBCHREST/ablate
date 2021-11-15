#ifndef ABLATELIBRARY_RIEMANN2GAS_HPP
#define ABLATELIBRARY_RIEMANN2GAS_HPP
#include <eos/eos.hpp>
#include <memory>
#include "fluxCalculator.hpp"

namespace ablate::finiteVolume::fluxCalculator {
/*
 * Computes the flux by treating all surfaces as Riemann problems, different perfect gas on left/right.
 */
class Riemann2Gas : public fluxCalculator::FluxCalculator {
   private:
    static Direction Riemann2GasFluxFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux,
                                        PetscReal *p12);
    PetscReal gammaVec[2];

   public:
    FluxCalculatorFunction GetFluxCalculatorFunction() override { return Riemann2GasFluxFunction; }
    void *GetFluxCalculatorContext() override { return (void *)&gammaVec; }
    explicit Riemann2Gas(std::shared_ptr<eos::EOS> eosL, std::shared_ptr<eos::EOS> eosR);
};
} // namespace ablate::finiteVolume::fluxCalculator

#endif  // ABLATELIBRARY_RIEMANN2GAS_HPP


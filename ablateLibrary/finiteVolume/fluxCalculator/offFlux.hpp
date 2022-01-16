#ifndef ABLATELIBRARY_OFFFLUX_HPP
#define ABLATELIBRARY_OFFFLUX_HPP
#include "fluxCalculator.hpp"
namespace ablate::finiteVolume::fluxCalculator {

/**
 * Turns off all flow through the flux calculator.  This is good for testing.
 */
class OffFlux : public fluxCalculator::FluxCalculator {
   private:
    static Direction OffCalculatorFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal pgsAlpha,
                                           PetscReal *massFlux, PetscReal *p12);

   public:
    FluxCalculatorFunction GetFluxCalculatorFunction() override { return OffCalculatorFunction; }
};
}  // namespace ablate::finiteVolume::fluxCalculator

#endif  // ABLATELIBRARY_OFFFLUX_HPP

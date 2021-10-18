#ifndef ABLATELIBRARY_AVERAGEFLUX_HPP
#define ABLATELIBRARY_AVERAGEFLUX_HPP
#include "fluxCalculator.hpp"

namespace ablate::flow::fluxCalculator {

/*
 * Computes the average flux from each side.  This is useful for debugging.
 */
class AverageFlux : public fluxCalculator::FluxCalculator {
   private:
    static Direction AvgCalculatorFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux,
                                           PetscReal *p12);

   public:
    FluxCalculatorFunction GetFluxCalculatorFunction() override { return AvgCalculatorFunction; }
};
}  // namespace ablate::flow::fluxCalculator

#endif  // ABLATELIBRARY_AVERAGEFLUX_HPP

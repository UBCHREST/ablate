#ifndef ABLATELIBRARY_AVERAGEFLUXDIFFERENCER_HPP
#define ABLATELIBRARY_AVERAGEFLUXDIFFERENCER_HPP
#include "fluxDifferencer.hpp"

namespace ablate::flow::fluxDifferencer {

/*
 * Computes the average flux from each side.  This is useful for debugging.
 */
class AverageFluxDifferencer : public fluxDifferencer::FluxDifferencer {
   private:
    static Direction AvgDifferencerFunction(void*, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                       PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                       PetscReal * massFlux, PetscReal *p12);

   public:
    FluxDifferencerFunction GetFluxDifferencerFunction() override { return AvgDifferencerFunction; }
};
}  // namespace ablate::flow::fluxDifferencer

#endif  // ABLATELIBRARY_AVERAGEFLUXDIFFERENCER_HPP

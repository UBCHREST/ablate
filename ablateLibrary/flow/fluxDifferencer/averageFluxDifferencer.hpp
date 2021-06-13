#ifndef ABLATELIBRARY_AVERAGEFLUXDIFFERENCER_HPP
#define ABLATELIBRARY_AVERAGEFLUXDIFFERENCER_HPP
#include "fluxDifferencer.hpp"

namespace ablate::flow::fluxDifferencer {

/*
 * Computes the average flux from each side.  This is useful for debugging.
 */
class AverageFluxDifferencer : public fluxDifferencer::FluxDifferencer {
   private:
    static void AvgDifferencerFunction(PetscReal Mm, PetscReal* sPm, PetscReal* sMm, PetscReal Mp, PetscReal* sPp, PetscReal* sMp);

   public:
    FluxDifferencerFunction GetFluxDifferencerFunction() override { return AvgDifferencerFunction; }
};
}  // namespace ablate::flow::fluxDifferencer

#endif  // ABLATELIBRARY_AVERAGEFLUXDIFFERENCER_HPP

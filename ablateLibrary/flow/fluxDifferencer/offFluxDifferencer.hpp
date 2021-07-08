#ifndef ABLATELIBRARY_OFFFLUXDIFFERENCER_HPP
#define ABLATELIBRARY_OFFFLUXDIFFERENCER_HPP
#include "fluxDifferencer.hpp"
namespace ablate::flow::fluxDifferencer {

/**
 * Turns off all flow through the flux differencer.  This is good for testing.
 */
class OffFluxDifferencer : public fluxDifferencer::FluxDifferencer {
   private:
    static Direction OffDifferencerFunction(void*, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                       PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                       PetscReal * massFlux, PetscReal *p12);


   public:
    FluxDifferencerFunction GetFluxDifferencerFunction() override { return OffDifferencerFunction; }
};
}  // namespace ablate::flow::fluxDifferencer

#endif  // ABLATELIBRARY_OFFFLUXDIFFERENCER_HPP

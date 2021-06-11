#ifndef ABLATELIBRARY_OFFFLUXDIFFERENCER_HPP
#define ABLATELIBRARY_OFFFLUXDIFFERENCER_HPP
#include "fluxDifferencer.hpp"
namespace ablate::flow::fluxDifferencer {

/**
 * Turns off all flow through the flux differencer.  This is good for testing.
 */
class OffFluxDifferencer : public fluxDifferencer::FluxDifferencer {
   private:
    static void OffDifferencerFunction(PetscReal Mm, PetscReal* sPm, PetscReal* sMm, PetscReal Mp, PetscReal* sPp, PetscReal* sMp);

   public:
    FluxDifferencerFunction GetFluxDifferencerFunction() override { return OffDifferencerFunction; }
};
}

#endif  // ABLATELIBRARY_OFFFLUXDIFFERENCER_HPP

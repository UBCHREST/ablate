#ifndef ABLATELIBRARY_AUSMFLUXDIFFERENCER_HPP
#define ABLATELIBRARY_AUSMFLUXDIFFERENCER_HPP
#include "fluxDifferencer.hpp"

namespace ablate::flow::fluxDifferencer {

/* Computes the min/plus values..
 * sPm: minus split pressure (P-), Capital script P in reference
 * sMm: minus split Mach Number (M-), Capital script M in reference
 * sPp: plus split pressure (P+), Capital script P in reference
 * sMp: plus split Mach Number (M+), Capital script M in reference
 */
class AusmFluxDifferencer : public fluxDifferencer::FluxDifferencer {
   private:
    static void AusmFluxDifferencerFunction(PetscReal Mm, PetscReal* sPm, PetscReal* sMm, PetscReal Mp, PetscReal* sPp, PetscReal* sMp);

   public:
    AusmFluxDifferencer() = default;
    AusmFluxDifferencer(AusmFluxDifferencer const&) = delete;
    AusmFluxDifferencer& operator=(AusmFluxDifferencer const&) = delete;
    ~AusmFluxDifferencer() override = default;

    FluxDifferencerFunction GetFluxDifferencerFunction() override { return AusmFluxDifferencerFunction; }
};
}  // namespace ablate::flow::fluxDifferencer
#endif  // ABLATELIBRARY_AUSMFLUXDIFFERENCER_HPP

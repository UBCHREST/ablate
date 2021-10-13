#ifndef ABLATELIBRARY_AUSM_HPP
#define ABLATELIBRARY_AUSM_HPP
#include "fluxCalculator.hpp"

namespace ablate::flow::fluxCalculator {

/* Computes the min/plus values..
 * sPm: minus split pressure (P-), Capital script P in reference
 * sMm: minus split Mach Number (M-), Capital script M in reference
 * sPp: plus split pressure (P+), Capital script P in reference
 * sMp: plus split Mach Number (M+), Capital script M in reference
 */
class Ausm : public fluxCalculator::FluxCalculator {
   private:
    static Direction AusmFunction(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* massFlux, PetscReal* p12);

   public:
    Ausm() = default;
    Ausm(Ausm const&) = delete;
    Ausm& operator=(Ausm const&) = delete;
    ~Ausm() override = default;

    FluxCalculatorFunction GetFluxCalculatorFunction() override { return AusmFunction; }
};
}  // namespace ablate::flow::fluxCalculator
#endif  // ABLATELIBRARY_AUSM_HPP

#ifndef ABLATELIBRARY_RIEMANN_H
#define ABLATELIBRARY_RIEMANN_H
#include <eos/eos.hpp>
#include <memory>
#include "fluxCalculator.hpp"

namespace ablate::finiteVolume::fluxCalculator {

/*
 * Computes the flux by treating all surfaces as Rieman problems.
 */
class Rieman : public fluxCalculator::FluxCalculator {
   private:
    static Direction RiemanFluxFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux,
                                        PetscReal *p12);
    PetscReal gamma;

   public:
    FluxCalculatorFunction GetFluxCalculatorFunction() override { return RiemanFluxFunction; }
    void *GetFluxCalculatorContext() override { return (void *)&gamma; }
    explicit Rieman(std::shared_ptr<eos::EOS> eos);
};
}  // namespace ablate::finiteVolume::fluxCalculator

#endif  // ABLATELIBRARY_RIEMANN_H

#ifndef ABLATELIBRARY_RIEMANN_H
#define ABLATELIBRARY_RIEMANN_H
#include <eos/eos.hpp>
#include <memory>
#include "riemannSolver.hpp"

namespace ablate::finiteVolume::fluxCalculator {

/*
 * Computes the flux by treating all surfaces as Rieman problems.
 */
class Riemann : public RiemannSolver {
   private:
    static Direction RiemannFluxFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux,
                                        PetscReal *p12);
    PetscReal gamma;

   public:
    FluxCalculatorFunction GetFluxCalculatorFunction() override { return RiemannFluxFunction; }
    void *GetFluxCalculatorContext() override { return (void *)&gamma; }
    explicit Riemann(std::shared_ptr<eos::EOS> eos);
};
}  // namespace ablate::finiteVolume::fluxCalculator

#endif  // ABLATELIBRARY_RIEMANN_H

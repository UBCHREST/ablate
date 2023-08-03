#ifndef ABLATELIBRARY_RIEMANNSTIFFR_HPP
#define ABLATELIBRARY_RIEMANNSTIFFR_HPP
#include <eos/eos.hpp>
#include <memory>
#include "riemannSolver.hpp"

namespace ablate::finiteVolume::fluxCalculator {
/*
 * Computes the flux by treating all surfaces as Riemann problems, different stiffened gas on left/right.
 * Reference Chang and Liou, JCP, 2007, Appendix B
 */
class RiemannStiff : public fluxCalculator::RiemannSolver {
   private:
    static Direction RiemannStiffFluxFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux,
                                              PetscReal *p12);
    PetscReal gammaVec[4];  // gamma_L, gamma_R, p0_L, p0_R

   public:
    FluxCalculatorFunction GetFluxCalculatorFunction() override { return RiemannStiffFluxFunction; }
    void *GetFluxCalculatorContext() override { return (void *)&gammaVec; }
    explicit RiemannStiff(std::shared_ptr<eos::EOS> eosL, std::shared_ptr<eos::EOS> eosR);
};

}  // namespace ablate::finiteVolume::fluxCalculator




#endif  // ABLATELIBRARY_RIEMANNSTIFFR_HPP

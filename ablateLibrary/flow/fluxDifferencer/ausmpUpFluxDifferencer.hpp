#ifndef ABLATELIBRARY_AUSMPUPFLUXDIFFERENCER_HPP
#define ABLATELIBRARY_AUSMPUPFLUXDIFFERENCER_HPP

#include "fluxDifferencer.hpp"
namespace ablate::flow::fluxDifferencer {

class AusmpUpFluxDifferencer : public fluxDifferencer::FluxDifferencer {
   private:
    static void AusmpUpFluxDifferencerFunction(void*, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                               PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                               PetscReal * massFlux, PetscReal *p12);

    static PetscReal M1Plus(PetscReal m);
    static PetscReal M2Plus(PetscReal m);
    static PetscReal M1Minus(PetscReal m);
    static PetscReal M2Minus(PetscReal m);
    const inline static PetscReal beta = 1.e+0 / 8.e+0;
    const inline static PetscReal alpha = 3.e+0 / 16.e+0;
    const inline static PetscReal Kp = 0.25;
    const inline static PetscReal Ku = 0.75;
    const inline static PetscReal sigma = 0.25;

   public:
    AusmpUpFluxDifferencer() = default;
    AusmpUpFluxDifferencer(AusmpUpFluxDifferencer const&) = delete;
    AusmpUpFluxDifferencer& operator=(AusmpUpFluxDifferencer const&) = delete;
    ~AusmpUpFluxDifferencer() override = default;

    FluxDifferencerFunction GetFluxDifferencerFunction() override { return AusmpUpFluxDifferencerFunction; }


    /**
     * Support calls
     * @param m
     * @return
     */
    static PetscReal M4Plus(PetscReal m);
    static PetscReal M4Minus(PetscReal m);
    static PetscReal P5Plus(PetscReal m);
    static PetscReal P5Minus(PetscReal m);


};

}
#endif  // ABLATELIBRARY_AUSMPUPFLUXDIFFERENCER_HPP

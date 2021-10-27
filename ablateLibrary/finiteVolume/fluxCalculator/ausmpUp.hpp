#ifndef ABLATELIBRARY_AUSMPUP_HPP
#define ABLATELIBRARY_AUSMPUP_HPP

#include "fluxCalculator.hpp"
namespace ablate::finiteVolume::fluxCalculator {

/**
 * A sequel to AUSM, Part II: AUSM+-up for all speeds
 */
class AusmpUp : public fluxCalculator::FluxCalculator {
   private:
    static Direction AusmpUpFunction(void*, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* massFlux, PetscReal* p12);

    static PetscReal M1Plus(PetscReal m);
    static PetscReal M2Plus(PetscReal m);
    static PetscReal M1Minus(PetscReal m);
    static PetscReal M2Minus(PetscReal m);
    const inline static PetscReal beta = 1.e+0 / 8.e+0;
    const inline static PetscReal Kp = 0.25;
    const inline static PetscReal Ku = 0.75;
    const inline static PetscReal sigma = 1.0;

    // The reference infinity mach number
    const double mInf;

   public:
    explicit AusmpUp(double mInf);
    AusmpUp(AusmpUp const&) = delete;
    AusmpUp& operator=(AusmpUp const&) = delete;
    ~AusmpUp() override = default;

    FluxCalculatorFunction GetFluxCalculatorFunction() override { return AusmpUpFunction; }
    void* GetFluxCalculatorContext() override { return (void*)&mInf; }

    /**
     * Support calls
     * @param m
     * @return
     */
    static PetscReal M4Plus(PetscReal m);
    static PetscReal M4Minus(PetscReal m);
    static PetscReal P5Plus(PetscReal m, double fa);
    static PetscReal P5Minus(PetscReal m, double fa);
};

}  // namespace ablate::finiteVolume::fluxCalculator
#endif  // ABLATELIBRARY_AUSMPUP_HPP

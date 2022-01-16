#ifndef ABLATELIBRARY_FLUXCALCULATOR_HPP
#define ABLATELIBRARY_FLUXCALCULATOR_HPP
#include <petsc.h>

namespace ablate::finiteVolume::fluxCalculator {

/**
 * This function returns the flow direction
 * > 0 left to right
 * < 0 right to left
 */
enum Direction { LEFT = 1, RIGHT = 2, NA = 0 };
using FluxCalculatorFunction = Direction (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal pgsAlpha,
                                             PetscReal* massFlux, PetscReal* p12);

class FluxCalculator {
   public:
    FluxCalculator() = default;
    FluxCalculator(FluxCalculator const&) = delete;
    FluxCalculator& operator=(FluxCalculator const&) = delete;
    virtual ~FluxCalculator() = default;
    virtual FluxCalculatorFunction GetFluxCalculatorFunction() = 0;
    virtual void* GetFluxCalculatorContext() { return nullptr; }
};
}  // namespace ablate::finiteVolume::fluxCalculator
#endif  // ABLATELIBRARY_FLUXCALCULATOR_HPP

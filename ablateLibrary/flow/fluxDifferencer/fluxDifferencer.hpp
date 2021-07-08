#ifndef ABLATELIBRARY_FLUXDIFFERENCER_HPP
#define ABLATELIBRARY_FLUXDIFFERENCER_HPP
#include <petsc.h>

namespace ablate::flow::fluxDifferencer {

/**
 * This function returns the flow direction
 * > 0 left to right
 * < 0 right to left
 */
enum Direction { LEFT = 1, RIGHT = 2, NA = 0 };
using FluxDifferencerFunction = Direction (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal* massFlux,
                                              PetscReal* p12);

class FluxDifferencer {
   public:
    FluxDifferencer() = default;
    FluxDifferencer(FluxDifferencer const&) = delete;
    FluxDifferencer& operator=(FluxDifferencer const&) = delete;
    virtual ~FluxDifferencer() = default;
    virtual FluxDifferencerFunction GetFluxDifferencerFunction() = 0;
    virtual void* GetFluxDifferencerContext() { return nullptr; }
};
}  // namespace ablate::flow::fluxDifferencer
#endif  // ABLATELIBRARY_FLUXDIFFERENCER_HPP

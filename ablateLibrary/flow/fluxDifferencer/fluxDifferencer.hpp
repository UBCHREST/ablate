#ifndef ABLATELIBRARY_FLUXDIFFERENCER_HPP
#define ABLATELIBRARY_FLUXDIFFERENCER_HPP
#include <petsc.h>

namespace ablate::flow::fluxDifferencer {

using FluxDifferencerFunction = void (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                         PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                         PetscReal * massFlux, PetscReal *p12);

class FluxDifferencer {
   public:
    FluxDifferencer() = default;
    FluxDifferencer(FluxDifferencer const&) = delete;
    FluxDifferencer& operator=(FluxDifferencer const&) = delete;
    virtual ~FluxDifferencer() = default;
    virtual FluxDifferencerFunction GetFluxDifferencerFunction() = 0;
};
}  // namespace ablate::flow::fluxDifferencer
#endif  // ABLATELIBRARY_FLUXDIFFERENCER_HPP

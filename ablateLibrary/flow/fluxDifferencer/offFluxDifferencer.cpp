#include "offFluxDifferencer.hpp"
ablate::flow::fluxDifferencer::Direction ablate::flow::fluxDifferencer::OffFluxDifferencer::OffDifferencerFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR,
                                                                                                                   PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux, PetscReal *p12) {
    *massFlux = 0.0;
    if (p12) {
        *p12 = 0.0;
    }
    return NA;
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::flow::fluxDifferencer::FluxDifferencer, ablate::flow::fluxDifferencer::OffFluxDifferencer, "Turns of convective flux through the face.");
#include "ausmFluxDifferencer.hpp"
ablate::flow::fluxDifferencer::Direction ablate::flow::fluxDifferencer::AusmFluxDifferencer::AusmFluxDifferencerFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR,
                                                                                                                         PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux,
                                                                                                                         PetscReal *p12) {
    PetscReal Mm = uR / aR;
    PetscReal sMm, sPm;
    if (PetscAbsReal(Mm) <= 1.) {
        sMm = -0.25 * PetscSqr(Mm - 1);
        sPm = -(sMm) * (2 + Mm);
    } else {
        sMm = 0.5 * (Mm - PetscAbsReal(Mm));
        sPm = (sMm) / Mm;
    }

    PetscReal Mp = uL / aL;
    PetscReal sMp, sPp;
    if (PetscAbsReal(Mp) <= 1.) {
        sMp = 0.25 * PetscSqr(Mp + 1);
        sPp = (sMp) * (2 - Mp);
    } else {
        sMp = 0.5 * (Mp + PetscAbsReal(Mp));
        sPp = (sMp) / Mp;
    }

    // compute the combined M
    PetscReal m = sMm + sMp;

    Direction dir;
    if (m < 0) {
        // M- on Right
        *massFlux = m * aR * rhoR;
        dir = RIGHT;
    } else {
        // M+ on Left
        *massFlux = m * aL * rhoL;
        dir = LEFT;
    }

    if (p12) {
        *p12 = pR * sPm + pL * sPp;
    }

    return dir;
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::flow::fluxDifferencer::FluxDifferencer, ablate::flow::fluxDifferencer::AusmFluxDifferencer,
                           "AUSM Flux Spliting: \"A New Flux Splitting Scheme\" Liou and Steffen, pg 26, Eqn (6), 1993");
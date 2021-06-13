#include "ausmFluxDifferencer.hpp"
void ablate::flow::fluxDifferencer::AusmFluxDifferencer::AusmFluxDifferencerFunction(PetscReal Mm, PetscReal *sPm, PetscReal *sMm, PetscReal Mp, PetscReal *sPp, PetscReal *sMp) {
    if (PetscAbsReal(Mm) <= 1.) {
        *sMm = -0.25 * PetscSqr(Mm - 1);
        *sPm = -(*sMm) * (2 + Mm);
    } else {
        *sMm = 0.5 * (Mm - PetscAbsReal(Mm));
        *sPm = (*sMm) / Mm;
    }
    if (PetscAbsReal(Mp) <= 1.) {
        *sMp = 0.25 * PetscSqr(Mp + 1);
        *sPp = (*sMp) * (2 - Mp);
    } else {
        *sMp = 0.5 * (Mp + PetscAbsReal(Mp));
        *sPp = (*sMp) / Mp;
    }

    // compute the combined M
    PetscReal m = *sMm + *sMp;

    if (m < 0) {
        // M- on Right
        *sMm = m;
        *sMp = 0.0;  // Zero out the left contribution
    } else {
        // M+ on Left
        *sMm = 0.0;
        *sMp = m;  // Zero out the right contribution
    }
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::flow::fluxDifferencer::FluxDifferencer, ablate::flow::fluxDifferencer::AusmFluxDifferencer,
                           "AUSM Flux Spliting: \"A New Flux Splitting Scheme\" Liou and Steffen, pg 26, Eqn (6), 1993");
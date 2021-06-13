#include "offFluxDifferencer.hpp"
void ablate::flow::fluxDifferencer::OffFluxDifferencer::OffDifferencerFunction(PetscReal Mm, PetscReal *sPm, PetscReal *sMm, PetscReal Mp, PetscReal *sPp, PetscReal *sMp) {
    *sPm = 0.0;
    *sPp = 0.0;
    *sMm = 0.0;
    *sMp = 0.0;
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::flow::fluxDifferencer::FluxDifferencer, ablate::flow::fluxDifferencer::OffFluxDifferencer, "Turns of convective flux through the face.");
#include "averageFluxDifferencer.hpp"
void ablate::flow::fluxDifferencer::AverageFluxDifferencer::AvgDifferencerFunction(PetscReal Mm, PetscReal *sPm, PetscReal *sMm, PetscReal Mp, PetscReal *sPp, PetscReal *sMp) {
    *sPm = 0.5;
    *sPp = 0.5;
    *sMm = Mm/2.0;
    *sMp = Mp/2.0;
}

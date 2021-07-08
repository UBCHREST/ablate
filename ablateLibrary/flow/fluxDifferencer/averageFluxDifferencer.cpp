#include "averageFluxDifferencer.hpp"
ablate::flow::fluxDifferencer::Direction ablate::flow::fluxDifferencer::AverageFluxDifferencer::AvgDifferencerFunction(void*, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
                                                                                   PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
                                                                                   PetscReal * massFlux, PetscReal *p12) {

    *massFlux = 0.5 * (uL*rhoL + uR*rhoR);

    if(p12){
        *p12 = 0.5 * (pL + pR);
    }
    return NA;
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::flow::fluxDifferencer::FluxDifferencer, ablate::flow::fluxDifferencer::AverageFluxDifferencer,
                           "Takes the average of the left/right faces.  Only useful for debugging.");
#include "averageFlux.hpp"
ablate::flow::fluxCalculator::Direction ablate::flow::fluxCalculator::AverageFlux::AvgCalculatorFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR, PetscReal aR,
                                                                                                         PetscReal rhoR, PetscReal pR, PetscReal *massFlux, PetscReal *p12) {
    *massFlux = 0.5 * (uL * rhoL + uR * rhoR);

    if (p12) {
        *p12 = 0.5 * (pL + pR);
    }
    return NA;
}

#include "parser/registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::flow::fluxCalculator::FluxCalculator, ablate::flow::fluxCalculator::AverageFlux, "Takes the average of the left/right faces.  Only useful for debugging.");
#include "offFlux.hpp"
ablate::finiteVolume::fluxCalculator::Direction ablate::finiteVolume::fluxCalculator::OffFlux::OffCalculatorFunction(void *, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL, PetscReal uR,
                                                                                                                     PetscReal aR, PetscReal rhoR, PetscReal pR, PetscReal *massFlux, PetscReal *p12) {
    *massFlux = 0.0;
    if (p12) {
        *p12 = 0.0;
    }
    return NA;
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::finiteVolume::fluxCalculator::FluxCalculator, ablate::finiteVolume::fluxCalculator::OffFlux, "Turns of convective flux through the face.");
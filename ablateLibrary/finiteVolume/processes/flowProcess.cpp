#include "flowProcess.hpp"

void ablate::finiteVolume::processes::FlowProcess::DecodeEulerState(eos::DecodeStateFunction decodeStateFunction, void *decodeStateContext, PetscInt dim, const PetscReal *conservedValues,
                                                                    const PetscReal *densityYi, const PetscReal *normal, PetscReal *density, PetscReal *normalVelocity, PetscReal *velocity,
                                                                    PetscReal *internalEnergy, PetscReal *a, PetscReal *M, PetscReal *p) {
    // decode
    *density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE] / (*density);

    // Get the velocity in this direction
    (*normalVelocity) = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[RHOU + d] / (*density);
        (*normalVelocity) += velocity[d] * normal[d];
    }

    // decode the state in the eos
    decodeStateFunction(dim, *density, totalEnergy, velocity, densityYi, internalEnergy, a, p, decodeStateContext);
    *M = (*normalVelocity) / (*a);
}

void ablate::finiteVolume::processes::FlowProcess::DecodeEulerState(eos::DecodeStateFunction decodeStateFunction, void *decodeStateContext, PetscInt dim, const PetscReal *conservedValues,
                                                                    const PetscReal *densityYi, PetscReal *density, PetscReal *velocity, PetscReal *internalEnergy, PetscReal *a, PetscReal *M,
                                                                    PetscReal *p) {
    // decode
    *density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE] / (*density);

    // Get the velocity in this direction
    PetscReal velMag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[RHOU + d] / (*density);
        velMag += velocity[d] * velocity[d];
    }

    // decode the state in the eos
    decodeStateFunction(dim, *density, totalEnergy, velocity, densityYi, internalEnergy, a, p, decodeStateContext);
    *M = PetscSqrtReal(velMag) / (*a);
}

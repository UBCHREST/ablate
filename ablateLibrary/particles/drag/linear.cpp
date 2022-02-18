#include "linear.hpp"

void ablate::particles::drag::Linear::ComputeDragForce(const PetscInt dim, const PetscReal *partVel, const PetscReal *flowVel, const PetscReal muF, const PetscReal rhoF, const PetscReal partDiam,
                                                       PetscReal *dragForce) {
    PetscReal dragForcePrefactor = -3.0 * PETSC_PI * partDiam * muF;

    for (int n = 0; n < dim; n++) {
        dragForce[n] = dragForcePrefactor * (partVel[n] - flowVel[n]);
    }
};

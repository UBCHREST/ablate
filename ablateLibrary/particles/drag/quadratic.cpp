#include "quadratic.hpp"
#include "utilities/mathUtilities.hpp"

void ablate::particles::drag::Quadratic::ComputeDragForce(const PetscInt dim, const PetscReal *partVel, const PetscReal *flowVel, const PetscReal muF, const PetscReal rhoF, const PetscReal partDiam, PetscReal *dragForce) {
    PetscReal* relVel[dim];
    PetscReal dragForcePrefactor = -0.44 * 0.5 * (PETSC_PI / 4.0) * partDiam * partDiam * rhoF;
    
    for (int n = 0; n < dim; n++) {
        *relVel[n]   = partVel[n] - flowVel[n];
        dragForce[n] = dragForcePrefactor * (partVel[n] - flowVel[n]) * ablate::utilities::MathUtilities::MagVector(dim, *relVel);
    }
};

// TODO: tests
// pure partVel, pure flowVel, both
// different dimensions: 1, 2, 3

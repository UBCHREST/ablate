#include "quadratic.hpp"
#include "utilities/mathUtilities.hpp"

void ablate::particles::drag::Quadratic::ComputeDragForce(const PetscInt dim, const PetscReal *partVel, const PetscReal *flowVel, const PetscReal muF, const PetscReal rhoF, const PetscReal partDiam, PetscReal *dragForce) {
    //const PetscReal relVel = *partVel - *flowVel;
    //PetscReal dragForcePrefactor = -0.44 * 0.5 * (PETSC_PI / 4.0) * partDiam * partDiam * rhoF * ablate::utilities::MathUtilities::MagVector(dim, *relVel);
    
    //for (int n = 0; n < dim; n++) {
        //dragForce[n] = dragForcePrefactor * (partVel[n] - flowVel[n]);
    //}
};

// TODO: tests
// pure partVel, pure flowVel, both
// different dimensions: 1, 2, 3

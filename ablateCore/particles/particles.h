#ifndef ABLATE_PARTICLES_H
#define ABLATE_PARTICLES_H

#include <petsc.h>
#include "flow.h"

struct _ParticleData {
    PetscBag parameters; /* constant particle parameters */
    DM dm;               /* particle domain */

    void* data; /* implementation-specific data */

    PetscReal timeInitial; /* The time for ui, at the beginning of the advection solve */
    PetscReal timeFinal;   /* The time for uf, at the end of the advection solve */
    Vec flowInitial;       /* The PDE solution field at ti */
    Vec flowFinal;         /* The PDE solution field at tf */
    Vec particleSolution;  /* The solution to the particles */
    Vec initialLocation;   /* The initial location of each particle */

    // Store the velocity field id in the flow
    PetscInt flowVelocityFieldIndex;

    // Allow the user to set the exactSolution
    PetscErrorCode (*exactSolution)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*);
    void* exactSolutionContext; /* the context for the exact solution */
};

typedef struct _ParticleData* ParticleData;

PetscErrorCode ParticleCreate(ParticleData* particles, PetscInt ndims);
PETSC_EXTERN PetscErrorCode ParticleInitializeFlow(ParticleData particles, FlowData flow);
PETSC_EXTERN PetscErrorCode ParticleSetExactSolutionFlow(ParticleData particles, PetscErrorCode (*exactSolution)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*),void* exactSolutionContext);
PetscErrorCode ParticleDestroy(ParticleData* particles);

#endif  // ABLATE_PARTICLES_H

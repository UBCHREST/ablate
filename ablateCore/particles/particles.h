#ifndef ABLATE_PARTICLES_H
#define ABLATE_PARTICLES_H

#include <petsc.h>
#include "particleInitializer.h"

// Define field id for mass
#define MASS 0

typedef const char* ParticleType;
#define PARTICLETRACER "particleTracer"

struct _Particles {
    PetscBag parameters; /* constant particle parameters */
    DM dm;               /* particle domain */
    DM flowDM;
    ParticleInitializer particleInitializer;

    void* data; /* implementation-specific data */

    PetscReal timeInitial; /* The time for ui, at the beginning of the advection solve */
    PetscReal timeFinal; /* The time for uf, at the end of the advection solve */
    Vec       flowInitial; /* The PDE solution field at ti */
    Vec       flowFinal; /* The PDE solution field at tf */
    Vec particleSolution; /* The solution to the particles */
    Vec initialLocation; /* The initial location of each particle */

    // Allow the user to set the exactSolution
    PetscErrorCode (*exactSolution)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
};

typedef struct _Particles* Particles;

PETSC_EXTERN PetscErrorCode ParticleCreate(Particles* particles, DM flowDM, ParticleInitializer particleInitializer);
PETSC_EXTERN PetscErrorCode ParticleTracerSetupIntegrator(Particles particles, TS particleTs, TS flowTs);
PETSC_EXTERN PetscErrorCode ParticleDestroy(Particles* particles);

#endif  // ABLATE_PARTICLES_H

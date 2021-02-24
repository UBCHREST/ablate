#ifndef ABLATE_PARTICLEINITIALIZER_H
#define ABLATE_PARTICLEINITIALIZER_H

#include <petsc.h>

typedef enum {PART_LAYOUT_CELL, PART_LAYOUT_BOX, NUM_PART_LAYOUT_TYPES} PartLayoutType;

struct _ParticleInitializer {
    PartLayoutType type;

    void* data; /* implementation-specific data */

    /* initialize the particle location/values*/
    PetscErrorCode (*initializeParticles)(struct _ParticleInitializer*, DM flowDm, DM particleDm);
    PetscErrorCode (*destroy)(struct _ParticleInitializer*);
};

typedef struct _ParticleInitializer* ParticleInitializer;

PETSC_EXTERN PetscErrorCode ParticleInitializerCreate(ParticleInitializer* particleInitializer);
PETSC_EXTERN PetscErrorCode ParticleInitializerDestroy(ParticleInitializer* particleInitializer);
PETSC_EXTERN PetscErrorCode ParticleInitialize(ParticleInitializer particleInitializer, DM flowDm, DM particleDm);


#endif  // ABLATE_PARTICLEINITIALIZER_H

#ifndef ABLATE_PARTICLEINITIALIZER_H
#define ABLATE_PARTICLEINITIALIZER_H

#include <petsc.h>

typedef enum {PART_LAYOUT_CELL, PART_LAYOUT_BOX, NUM_PART_LAYOUT_TYPES} PartLayoutType;

PETSC_EXTERN PetscErrorCode ParticleInitialize(DM flowDm, DM particleDm);

#endif  // ABLATE_PARTICLEINITIALIZER_H

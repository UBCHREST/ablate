#ifndef ABLATE_PARTICLETRACER_H
#define ABLATE_PARTICLETRACER_H
#include "particles.h"

PETSC_EXTERN PetscErrorCode ParticleTracerCreate(Particles* particles, DM flowDM, Vec flowField, ParticleInitializer particleInitializer);
PETSC_EXTERN PetscErrorCode ParticleTracerDestroy(Particles* particles);
#endif  // ABLATE_PARTICLETRACER_H

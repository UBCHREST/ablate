#ifndef ABLATE_PARTICLETRACER_H
#define ABLATE_PARTICLETRACER_H
#include "particles.h"

PETSC_EXTERN const char ParticleTracerVelocity[];

PETSC_EXTERN PetscErrorCode ParticleTracerCreate(ParticleData* particles, PetscInt ndims);
PETSC_EXTERN PetscErrorCode ParticleTracerSetupIntegrator(ParticleData particles, TS particleTs, FlowData flowTs);
PETSC_EXTERN PetscErrorCode ParticleTracerDestroy(ParticleData* particles);
#endif  // ABLATE_PARTICLETRACER_H

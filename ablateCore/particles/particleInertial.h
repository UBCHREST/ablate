#ifndef ABLATE_PARTICLEINERTIAL_H
#define ABLATE_PARTICLEINERTIAL_H
#include "particles.h"


PETSC_EXTERN PetscErrorCode ParticleInertialCreate(ParticleData* particles, PetscInt ndims);
PETSC_EXTERN PetscErrorCode ParticleInertialSetupIntegrator(ParticleData particles, TS particleTs, FlowData flowTs);
PETSC_EXTERN PetscErrorCode ParticleInertialDestroy(ParticleData* particles);
#endif  // ABLATE_PARTICLEINERTIAL_H

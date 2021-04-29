#ifndef ABLATE_PARTICLEINERTIAL_H
#define ABLATE_PARTICLEINERTIAL_H
#include "particles.h"


typedef struct {
    PetscReal fluidDensity; // fluid density needed for particle drag force
    PetscReal fluidViscosity; // fluid viscosity needed for particle drag force
    PetscReal gravityField[3]; //gravity field
}InertialParticleParameters;

PETSC_EXTERN PetscErrorCode ParticleInertialCreate(ParticleData* particles, PetscInt ndims);
PETSC_EXTERN PetscErrorCode ParticleInertialSetupIntegrator(ParticleData particles, TS particleTs, FlowData flowTs);
PETSC_EXTERN PetscErrorCode ParticleInertialDestroy(ParticleData* particles);
#endif  // ABLATE_PARTICLEINERTIAL_H

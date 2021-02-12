#ifndef ABLATE_PARTICLECELLINITIALIZER_H
#define ABLATE_PARTICLECELLINITIALIZER_H

#include <petsc.h>
#include "particleInitializer.h"

PETSC_EXTERN PetscErrorCode ParticleCellInitializerCreate(ParticleInitializer particleInitializer);

#endif  // ABLATE_PARTICLECELLINITIALIZER_H
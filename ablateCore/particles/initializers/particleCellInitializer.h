#ifndef ABLATE_PARTICLECELLINITIALIZER_H
#define ABLATE_PARTICLECELLINITIALIZER_H

#include <petsc.h>
#include "particleInitializer.h"

PETSC_EXTERN PetscErrorCode ParticleCellInitialize(DM flowDm, DM particleDm);

#endif  // ABLATE_PARTICLECELLINITIALIZER_H
#ifndef ABLATE_PARTICLEBOXINITIALIZER_H
#define ABLATE_PARTICLEBOXINITIALIZER_H

#include <petsc.h>
#include "particleInitializer.h"

PETSC_EXTERN PetscErrorCode ParticleBoxInitializerCreate(ParticleInitializer particleInitializer);

#endif  // ABLATE_PARTICLEBOXINITIALIZER_H
#ifndef ABLATE_PARTICLEBOXINITIALIZER_H
#define ABLATE_PARTICLEBOXINITIALIZER_H

#include <petsc.h>
#include "particleInitializer.h"

PETSC_EXTERN PetscErrorCode ParticleBoxInitialize(DM flowDm, DM particleDm);

#endif  // ABLATE_PARTICLEBOXINITIALIZER_H
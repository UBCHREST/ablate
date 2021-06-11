#ifndef ABLATE_FLUXDIFFERENCER_H
#define ABLATE_FLUXDIFFERENCER_H
#include <petsc.h>

typedef void (*FluxDifferencerFunction)(PetscReal Mm, PetscReal* sPm, PetscReal* sMm,
                                            PetscReal Mp, PetscReal* sPp, PetscReal *sMp);

#endif
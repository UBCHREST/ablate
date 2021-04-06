#ifndef ABLATE_FLUXDIFFERENCER_H
#define ABLATE_FLUXDIFFERENCER_H
#include <petsc.h>
#include "flow.h"

typedef void (*FluxDifferencerFunction)(PetscReal Mm, PetscReal* sPm, PetscReal* sMm,
                                            PetscReal Mp, PetscReal* sPp, PetscReal *sMp);

PETSC_EXTERN PetscErrorCode FluxDifferencerRegister(const char*,const FluxDifferencerFunction);
PETSC_EXTERN PetscErrorCode FluxDifferencerGet(const char*, FluxDifferencerFunction* );
PETSC_EXTERN PetscErrorCode FluxDifferencerListGet(PetscFunctionList* list);
#endif
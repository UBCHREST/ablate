#ifndef ABLATE_PARTICLES_H
#define ABLATE_PARTICLES_H

#include <petsc.h>
#include "../parameters.h"

struct _Particles {
    PetscBag parameters; /* constant particle parameters */
    DM dm;               /* particle domain */
    Vec flowField;       /* flow solution vector */

    void* data; /* implementation-specific data */

    /* flow field setup parameters */
    PetscErrorCode (*setupDiscretization)(struct _Flow*);
    PetscErrorCode (*startProblemSetup)(struct _Flow*);
    PetscErrorCode (*completeProblemSetup)(struct _Flow*, TS ts);
    PetscErrorCode (*completeFlowInitialization)(DM dm, Vec u);
    PetscErrorCode (*destroy)(struct _Flow*);
};

typedef struct _Particles* Particles;

PETSC_EXTERN PetscErrorCode ParticleCreate(Particles* particles, Flow flow);
PETSC_EXTERN PetscErrorCode FlowDestroy(Flow* flow);

#endif  // ABLATE_PARTICLES_H

#ifndef ABLATE_PARTICLES_H
#define ABLATE_PARTICLES_H

#include <petsc.h>
#include "flow.h"

PETSC_EXTERN const char ParticleVelocity[];
PETSC_EXTERN const char ParticleDiameter[];
PETSC_EXTERN const char ParticleDensity[];

typedef struct {
    const char* fieldName;
    PetscInt components;
    PetscDataType type;
} ParticleFieldDescriptor;


struct _ParticleData {
    PetscBag parameters; /* constant particle parameters */
    DM dm;               /* particle domain */

    void* data; /* implementation-specific data */

    PetscReal timeInitial; /* The time for ui, at the beginning of the advection solve */
    PetscReal timeFinal;   /* The time for uf, at the end of the advection solve */
    Vec flowInitial;       /* The PDE solution field at ti */
    Vec flowFinal;         /* The PDE solution field at tf */

    // Store the velocity field id in the flow
    PetscInt flowVelocityFieldIndex;

    // Allow the user to set the exactSolution
    PetscErrorCode (*exactSolution)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*);
    void* exactSolutionContext; /* the context for the exact solution */

    /** store the fields on the dm swarm **/
    PetscInt numberFields;
    ParticleFieldDescriptor *fieldDescriptors;

    // store a boolean to state if a dmChanged (number of particles local/global changed)
    PetscBool dmChanged;
};

typedef struct _ParticleData* ParticleData;

PetscErrorCode ParticleCreate(ParticleData* particles, PetscInt ndims);
PETSC_EXTERN PetscErrorCode ParticleRegisterPetscDatatypeField(ParticleData particles, const char fieldname[],PetscInt blocksize,PetscDataType type);
PETSC_EXTERN PetscErrorCode ParticleInitializeFlow(ParticleData particles, FlowData flow);
PETSC_EXTERN PetscErrorCode ParticleSetExactSolutionFlow(ParticleData particles, PetscErrorCode (*exactSolution)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*),void* exactSolutionContext);
PETSC_EXTERN PetscErrorCode ParticleView(ParticleData particleData,PetscViewer viewer);
PETSC_EXTERN PetscErrorCode ParticleViewFromOptions(ParticleData particleData,PetscObject obj, char *title);
PetscErrorCode ParticleDestroy(ParticleData* particles);

#endif  // ABLATE_PARTICLES_H

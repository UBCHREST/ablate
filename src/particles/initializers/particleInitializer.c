#include "particleInitializer.h"
#include "particleCellInitializer.h"
#include "particleBoxInitializer.h"

const char *partLayoutTypes[NUM_PART_LAYOUT_TYPES+1] = {"cell", "box",  "unknown"};

PetscErrorCode ParticleInitializerCreate(ParticleInitializer* particleInitializer) {
    PetscErrorCode ierr;
    *particleInitializer = malloc(sizeof(struct _ParticleInitializer));

    // initialize all fields
    (*particleInitializer)->data = NULL;

    // determine type of initializer
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Particle Initialization Options", NULL);CHKERRQ(ierr);
    (*particleInitializer)->type = PART_LAYOUT_CELL;
    PetscInt selectedLayoutType = (*particleInitializer)->type;
    ierr = PetscOptionsEList("-particle_layout_type", "The particle layout type", NULL, partLayoutTypes, NUM_PART_LAYOUT_TYPES, partLayoutTypes[(*particleInitializer)->type], &selectedLayoutType, NULL);CHKERRQ(ierr);
    (*particleInitializer)->type = (PartLayoutType)selectedLayoutType;
    ierr = PetscOptionsEnd();

    switch((*particleInitializer)->type){
        case PART_LAYOUT_CELL:
            return ParticleCellInitializerCreate(*particleInitializer);
        case PART_LAYOUT_BOX:
            return ParticleBoxInitializerCreate(*particleInitializer);
        default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown particle initialization type");
    }
}

PETSC_EXTERN PetscErrorCode ParticleInitializerDestroy(ParticleInitializer* particleInitializer) {
    PetscErrorCode ierr = (*particleInitializer)->destroy(*particleInitializer);CHKERRQ(ierr);
    free(*particleInitializer);
    particleInitializer = NULL;
    return 0;
}

PETSC_EXTERN PetscErrorCode ParticleInitializerSetSolutionVector(ParticleInitializer particleInitializer,DM dm, Vec solution) {
    return particleInitializer->setSolutionVector(particleInitializer, dm, solution);
}

;
PETSC_EXTERN PetscErrorCode ParticleInitialize(ParticleInitializer particleInitializer,DM flowDm, DM particleDm){
    PetscErrorCode ierr =  particleInitializer->initializeParticles(particleInitializer, flowDm, particleDm);CHKERRQ(ierr);
    return DMViewFromOptions(particleDm, NULL, "-dm_view");
}

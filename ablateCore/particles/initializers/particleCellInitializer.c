#include "particleCellInitializer.h"

typedef struct {
    PetscInt particlesPerCell;
} CellInitializerData;

static PetscErrorCode initializeParticles(ParticleInitializer particleInitializer, DM flowDm, DM particleDm) {
    PetscFunctionBeginUser;
    CellInitializerData* data = (CellInitializerData*)particleInitializer->data;

    PetscInt       cStart, cEnd;
    PetscErrorCode ierr = DMPlexGetHeightStratum(flowDm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMSwarmSetLocalSizes(particleDm, (cEnd - cStart) * data->particlesPerCell, 0);CHKERRQ(ierr);
    ierr = DMSetFromOptions(particleDm);CHKERRQ(ierr);

    // set the cell ids
    PetscInt       *cellid;
    ierr = DMSwarmGetField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
    for (PetscInt c = cStart; c < cEnd; ++c) {
        for (PetscInt p = 0; p < data->particlesPerCell; ++p) {
            const PetscInt n = c*data->particlesPerCell + p;
            cellid[n] = c;
        }
    }

    ierr = DMSwarmRestoreField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
    ierr = DMSwarmSetPointCoordinatesRandom(particleDm, data->particlesPerCell);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode destroy(ParticleInitializer particleInitializer) {
    PetscFunctionBeginUser;
    free(particleInitializer->data);
    PetscFunctionReturn(0);
}

PetscErrorCode ParticleCellInitializerCreate(ParticleInitializer particleInitializer) {
    PetscFunctionBeginUser;
    particleInitializer->destroy = destroy;
    particleInitializer->initializeParticles = initializeParticles;

    CellInitializerData* data = malloc(sizeof(CellInitializerData));
    data->particlesPerCell = 1;

    // determine values for cell initializer
    PetscErrorCode ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Cell Particle Initialization Options", NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-Npc", "The initial number of particles per cell", NULL, data->particlesPerCell, &data->particlesPerCell, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();

    particleInitializer->data = data;
    PetscFunctionReturn(0);
}

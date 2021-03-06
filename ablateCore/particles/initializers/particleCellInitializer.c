#include "particleCellInitializer.h"

PetscErrorCode ParticleCellInitialize(DM flowDm, DM particleDm) {
    PetscFunctionBeginUser;

    PetscInt particlesPerCell;

    // determine values for cell initializer
    PetscErrorCode ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Cell Particle Initialization Options", NULL);CHKERRQ(ierr);
            ierr = PetscOptionsInt("-Npc", "The initial number of particles per cell", NULL, particlesPerCell, &particlesPerCell, NULL);CHKERRQ(ierr);
            ierr = PetscOptionsEnd();


    PetscInt       cStart, cEnd;
    ierr = DMPlexGetHeightStratum(flowDm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMSwarmSetLocalSizes(particleDm, (cEnd - cStart) * particlesPerCell, 0);CHKERRQ(ierr);
    ierr = DMSetFromOptions(particleDm);CHKERRQ(ierr);

    // set the cell ids
    PetscInt       *cellid;
    ierr = DMSwarmGetField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
    for (PetscInt c = cStart; c < cEnd; ++c) {
        for (PetscInt p = 0; p < particlesPerCell; ++p) {
            const PetscInt n = c*particlesPerCell + p;
            cellid[n] = c;
        }
    }

    ierr = DMSwarmRestoreField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
    ierr = DMSwarmSetPointCoordinatesRandom(particleDm, particlesPerCell);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

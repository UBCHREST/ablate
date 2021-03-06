#include "particleInitializer.h"
#include "particleBoxInitializer.h"
#include "particleCellInitializer.h"

const char* partLayoutTypes[NUM_PART_LAYOUT_TYPES + 1] = {"cell", "box", "unknown"};

PetscErrorCode ParticleInitialize(DM flowDm, DM particleDm) {
    // determine type of initializer
    PartLayoutType type = PART_LAYOUT_CELL;
    PetscErrorCode ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Particle Initialization Options", NULL);CHKERRQ(ierr);
    PetscInt selectedLayoutType = type;
    ierr = PetscOptionsEList("-particle_layout_type", "The particle layout type", NULL, partLayoutTypes, NUM_PART_LAYOUT_TYPES, partLayoutTypes[type], &selectedLayoutType, NULL);CHKERRQ(ierr);
    type = (PartLayoutType)selectedLayoutType;
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    switch (type) {
        case PART_LAYOUT_CELL:
            return ParticleCellInitialize(flowDm, particleDm);
        case PART_LAYOUT_BOX:
            return ParticleBoxInitialize(flowDm, particleDm);
        default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown particle initialization type");
    }
}

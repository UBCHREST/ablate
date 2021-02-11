#include "particleBoxInitializer.h"

typedef struct {
    PetscReal      partLower[3]; /* Lower left corner of particle box */
    PetscReal      partUpper[3]; /* Upper right corner of particle box */
    PetscInt       Npb;          /* The initial number of particles per box dimension */
} BoxInitializerData;

static PetscErrorCode setSolutionVector(ParticleInitializer particleInitializer,DM dm, Vec u) {
    PetscScalar *coords;
    PetscReal x[3], dx[3];
    PetscInt n[3];
    PetscInt Np, dim, d, i, j, k;
    BoxInitializerData *data = (BoxInitializerData *)particleInitializer->data;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    Np = 1;
    for (d = 0; d < dim; ++d) {
        n[d] = data->Npb;
        dx[d] = (data->partUpper[d] - data->partLower[d]) / PetscMax(1, n[d] - 1);
        Np *= n[d];
    }
    ierr = VecGetArray(u, &coords);CHKERRQ(ierr);
    switch (dim) {
        case 2:
            x[0] = data->partLower[0];
            for (i = 0; i < n[0]; ++i, x[0] += dx[0]) {
                x[1] = data->partLower[1];
                for (j = 0; j < n[1]; ++j, x[1] += dx[1]) {
                    const PetscInt p = j * n[0] + i;
                    for (d = 0; d < dim; ++d) coords[p * dim + d] = x[d];
                }
            }
            break;
        case 3:
            x[0] = data->partLower[0];
            for (i = 0; i < n[0]; ++i, x[0] += dx[0]) {
                x[1] = data->partLower[1];
                for (j = 0; j < n[1]; ++j, x[1] += dx[1]) {
                    x[2] = data->partLower[2];
                    for (k = 0; k < n[2]; ++k, x[2] += dx[2]) {
                        const PetscInt p = (k * n[1] + j) * n[0] + i;
                        for (d = 0; d < dim; ++d) coords[p * dim + d] = x[d];
                    }
                }
            }
            break;
        default:
            SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Do not support particle layout in dimension %D", dim);
    }
    ierr = VecRestoreArray(u, &coords);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode initializeParticles(ParticleInitializer particleInitializer, DM flowDm, DM particleDm) {
    PetscFunctionBeginUser;
    BoxInitializerData* data = (BoxInitializerData*)particleInitializer->data;

    PetscInt dim;
    PetscErrorCode ierr = DMGetDimension(particleDm, &dim);CHKERRQ(ierr);

    PetscInt Np = 1;
    PetscInt       *cellid, n[3];
    PetscReal       x[3], dx[3];
    PetscScalar    *coords;
    MPI_Comm mpiComm;

    for (PetscInt d = 0; d < dim; ++d) {
        n[d]  = data->Npb;
        dx[d] = (data->partUpper[d] - data->partLower[d])/PetscMax(1, n[d] - 1);
        Np   *= n[d];
    }
    ierr = DMSwarmSetLocalSizes(particleDm, Np, 0);CHKERRQ(ierr);
    ierr = DMSetFromOptions(particleDm);CHKERRQ(ierr);
    ierr = DMSwarmGetField(particleDm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
    switch (dim) {
        case 2:
            x[0] = data->partLower[0];
            for (PetscInt i = 0; i < n[0]; ++i, x[0] += dx[0]) {
                x[1] = data->partLower[1];
                for (PetscInt j = 0; j < n[1]; ++j, x[1] += dx[1]) {
                    const PetscInt p = j*n[0] + i;
                    for (PetscInt d = 0; d < dim; ++d){
                        coords[p*dim + d] = x[d];
                    }
                }
            }
            break;
        case 3:
            x[0] = data->partLower[0];
            for (PetscInt i = 0; i < n[0]; ++i, x[0] += dx[0]) {
                x[1] = data->partLower[1];
                for (PetscInt j = 0; j < n[1]; ++j, x[1] += dx[1]) {
                    x[2] = data->partLower[2];
                    for (PetscInt k = 0; k < n[2]; ++k, x[2] += dx[2]) {
                        const PetscInt p = (k*n[1] + j)*n[0] + i;
                        for (PetscInt d = 0; d < dim; ++d){
                            coords[p*dim + d] = x[d];
                        }
                    }
                }
            }
            break;
        default: SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Do not support particle layout in dimension %D", dim);
    }
    ierr = DMSwarmRestoreField(particleDm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
    ierr = DMSwarmGetField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
    for (PetscInt p = 0; p < Np; ++p){
        cellid[p] = 0;
    }
    ierr = DMSwarmRestoreField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
    ierr = DMSwarmMigrate(particleDm, PETSC_TRUE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}



PetscErrorCode destroy(ParticleInitializer particleInitializer) {
    PetscFunctionBeginUser;
    free(particleInitializer->data);
    PetscFunctionReturn(0);
}

PetscErrorCode ParticleBoxInitializerCreate(ParticleInitializer particleInitializer) {
    PetscFunctionBeginUser;
    particleInitializer->destroy = destroy;
    particleInitializer->initializeParticles = initializeParticles;
    particleInitializer->setSolutionVector = setSolutionVector;

    BoxInitializerData* data = malloc(sizeof(BoxInitializerData));
    data->Npb = 1;
    data->partLower[0] = data->partLower[1] = data->partLower[2] = 0.;
    data->partUpper[0] = data->partUpper[1] = data->partUpper[2] = 1.;

    // determine values for cell initializer
    PetscErrorCode ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Box Particle Initialization Options", NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-Npb", "The initial number of particles per box dimension", NULL, data->Npb, &data->Npb, NULL);CHKERRQ(ierr);
    PetscInt n = 3;
    ierr = PetscOptionsRealArray("-particle_lower", "The lower left corner of the particle box", NULL, data->partLower, &n, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsRealArray("-particle_upper", "The upper right corner of the particle box", NULL, data->partUpper, &n, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();

    particleInitializer->data = data;
    PetscFunctionReturn(0);
}

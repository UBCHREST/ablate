#include "particleTracer.h"
#include <petsc/private/dmswarmimpl.h>
#include <petscdmswarm.h>

/* x_t = v

   Note that here we use the velocity field at t_{n+1} to advect the particles from
   t_n to t_{n+1}. If we use both of these fields, we could use Crank-Nicholson or
   the method of characteristics.
*/
static PetscErrorCode freeStreaming(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
    ParticleData particles = (ParticleData)ctx;
    Vec u = particles->flowFinal;
    DM sdm, dm, vdm;
    Vec vel, locvel, pvel;
    IS vis;
    DMInterpolationInfo ictx;
    const PetscScalar *coords, *v;
    PetscScalar *f;
    PetscInt vf[1] = {0};
    PetscInt dim, Np;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &sdm);CHKERRQ(ierr);
    ierr = DMSwarmGetCellDM(sdm, &dm);CHKERRQ(ierr);
    ierr = DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(sdm, &pvel);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(sdm, &Np);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    /* Get local velocity */
    ierr = DMCreateSubDM(dm, 1, vf, &vis, &vdm);CHKERRQ(ierr);
    ierr = VecGetSubVector(u, vis, &vel);CHKERRQ(ierr);
    ierr = DMGetLocalVector(vdm, &locvel);CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(vdm, PETSC_TRUE, locvel, particles->timeInitial, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(vdm, vel, INSERT_VALUES, locvel);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(vdm, vel, INSERT_VALUES, locvel);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(u, vis, &vel);CHKERRQ(ierr);
    ierr = ISDestroy(&vis);CHKERRQ(ierr);

    /* Interpolate velocity */
    ierr = DMInterpolationCreate(PETSC_COMM_SELF, &ictx);CHKERRQ(ierr);
    ierr = DMInterpolationSetDim(ictx, dim);CHKERRQ(ierr);
    ierr = DMInterpolationSetDof(ictx, dim);CHKERRQ(ierr);
    ierr = VecGetArrayRead(X, &coords);CHKERRQ(ierr);
    ierr = DMInterpolationAddPoints(ictx, Np, (PetscReal *)coords);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X, &coords);CHKERRQ(ierr);
    ierr = DMInterpolationSetUp(ictx, vdm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMInterpolationEvaluate(ictx, vdm, locvel, pvel);CHKERRQ(ierr);
    ierr = DMInterpolationDestroy(&ictx);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(vdm, &locvel);CHKERRQ(ierr);
    ierr = DMDestroy(&vdm);CHKERRQ(ierr);

    ierr = VecGetArray(F, &f);CHKERRQ(ierr);
    ierr = VecGetArrayRead(pvel, &v);CHKERRQ(ierr);
    ierr = PetscArraycpy(f, v, Np * dim);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(pvel, &v);CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(sdm, &pvel);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode PrintParticlesToFile(TS flowTS, DM sdm) {
    PetscInt timeStep;
    TSGetStepNumber(flowTS, &timeStep);
    const PetscScalar *coord;

    PetscInt dim;
    PetscErrorCode ierr = DMGetDimension(sdm, &dim);CHKERRQ(ierr);
    PetscInt Np;
    ierr = DMSwarmGetLocalSize(sdm, &Np);CHKERRQ(ierr);

    ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);CHKERRQ(ierr);

    char fileName[80];
    sprintf(fileName, "particles.%d.txt", timeStep);

    FILE *f = fopen(fileName, "w");
    switch (dim) {
        case 3:
            fprintf(f, "X Y Z i \n");
            break;
        case 2:
            fprintf(f, "X Y i \n");
            break;
        case 1:
            fprintf(f, "X i \n");
            break;
    }
    for (PetscInt p = 0; p < Np; p++) {
        for (int d = 0; d < dim; d++) {
            fprintf(f, "%f ", coord[(p * dim) + d]);
        }
        fprintf(f, "%d \n", p);
    }
    fclose(f);
    ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);CHKERRQ(ierr);
    return 0;
}

static PetscErrorCode advectParticles(TS ts) {
    TS sts;
    DM sdm;
    Vec p;
    ParticleData particles;
    PetscScalar *coord, *a;
    const PetscScalar *ca;
    PetscReal time;
    PetscInt n;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = PetscObjectQuery((PetscObject)ts, "_SwarmTS", (PetscObject *)&sts);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)ts, "_SwarmSol", (PetscObject *)&p);CHKERRQ(ierr);
    ierr = TSGetDM(sts, &sdm);CHKERRQ(ierr);
    ierr = TSGetRHSFunction(sts, NULL, NULL, (void **)&particles);CHKERRQ(ierr);
    ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);CHKERRQ(ierr);
    ierr = VecGetLocalSize(p, &n);CHKERRQ(ierr);
    ierr = VecGetArray(p, &a);CHKERRQ(ierr);
    ierr = PetscArraycpy(a, coord, n);CHKERRQ(ierr);
    ierr = VecRestoreArray(p, &a);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);CHKERRQ(ierr);


    // Set the start time for TSSolve
    ierr = TSSetTime(sts, particles->timeInitial);CHKERRQ(ierr);

    // Set the max end time based upon the flow end time
    ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
    ierr = TSSetMaxTime(sts, time);CHKERRQ(ierr);
    particles->timeFinal = time;

    // take the needed timesteps to get to the flow time
    ierr = TSSolve(sts, p);CHKERRQ(ierr);
    ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);CHKERRQ(ierr);
    ierr = VecGetLocalSize(p, &n);CHKERRQ(ierr);
    ierr = VecGetArrayRead(p, &ca);CHKERRQ(ierr);
    ierr = PetscArraycpy(coord, ca, n);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(p, &ca);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord);CHKERRQ(ierr);

    ierr = VecCopy(particles->flowFinal, particles->flowInitial);CHKERRQ(ierr);
    particles->timeInitial = particles->timeFinal;

    ierr = DMSwarmMigrate(sdm, PETSC_TRUE);CHKERRQ(ierr);

    // debug code until output is updated
    ierr = PrintParticlesToFile(ts, sdm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(sdm, NULL, "-dm_view");CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ParticleTracerSetupIntegrator(ParticleData particles, TS particleTs, TS flowTs) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ierr = TSSetDM(particleTs, particles->dm);CHKERRQ(ierr);
    ierr = TSSetProblemType(particleTs, TS_NONLINEAR);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(particleTs, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetApplicationContext(particleTs, particles);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(particleTs, NULL, freeStreaming, particles);CHKERRQ(ierr);

    // link the solution with the flowTS
    ierr = TSSetPostStep(flowTs, advectParticles);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)flowTs, "_SwarmTS", (PetscObject)particleTs);CHKERRQ(ierr);  // to else where
    ierr = DMSwarmVectorDefineField(particles->dm, DMSwarmPICField_coor);CHKERRQ(ierr);  // do else where
    ierr = DMCreateGlobalVector(particles->dm, &(particles->particleSolution));CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)flowTs, "_SwarmSol", (PetscObject)(particles->particleSolution));CHKERRQ(ierr);  // do else where

    // Set up the TS
    ierr = TSSetFromOptions(particleTs);CHKERRQ(ierr);

    // extract the initial solution
    Vec xtmp;
    ierr = DMCreateGlobalVector(particles->dm, &(particles->initialLocation));CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(particles->dm, DMSwarmPICField_coor, &xtmp);CHKERRQ(ierr);
    ierr = VecCopy(xtmp, particles->initialLocation);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(particles->dm, DMSwarmPICField_coor, &xtmp);CHKERRQ(ierr);

    // debug code until output is updated
    ierr = PrintParticlesToFile(flowTs, particles->dm);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ParticleTracerCreate(ParticleData *particles, PetscInt ndims) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Call the base particle create
    ierr = ParticleCreate(particles, ndims);CHKERRQ(ierr);

    // register all particle fields
    ierr = DMSwarmSetType((*particles)->dm, DMSWARM_PIC);CHKERRQ(ierr);
    ierr = DMSwarmRegisterPetscDatatypeField((*particles)->dm, "mass", 1, PETSC_REAL);CHKERRQ(ierr);
    ierr = DMSwarmFinalizeFieldRegister((*particles)->dm);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ParticleTracerDestroy(ParticleData *particles) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr = VecDestroy(&((*particles)->flowInitial));CHKERRQ(ierr);
    ierr = VecDestroy(&((*particles)->particleSolution));CHKERRQ(ierr);
    ierr = VecDestroy(&((*particles)->initialLocation));CHKERRQ(ierr);

    // Call the base destroy
    ierr = ParticleDestroy(particles);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

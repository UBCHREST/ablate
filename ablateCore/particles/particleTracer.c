#include "particleTracer.h"
#include <petsc/private/dmswarmimpl.h>
#include <petscdmswarm.h>

const char ParticleTracerVelocity[] = "ParticleTracerVelocity";

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
    PetscInt vf[1] = {particles->flowVelocityFieldIndex};
    PetscInt dim, Np;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &sdm);CHKERRQ(ierr);
    ierr = DMSwarmGetCellDM(sdm, &dm);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, ParticleTracerVelocity, &pvel);CHKERRQ(ierr);
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

    /* Particles that lie outside the domain should be dropped,
     whereas particles that move to another partition should trigger a migration */
    ierr = DMInterpolationSetUp(ictx, vdm, PETSC_FALSE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecSet(pvel, 0.);CHKERRQ(ierr);

    ierr = DMInterpolationEvaluate(ictx, vdm, locvel, pvel);CHKERRQ(ierr);
    ierr = DMInterpolationDestroy(&ictx);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(vdm, &locvel);CHKERRQ(ierr);
    ierr = DMDestroy(&vdm);CHKERRQ(ierr);

    ierr = VecGetArray(F, &f);CHKERRQ(ierr);
    ierr = VecGetArrayRead(pvel, &v);CHKERRQ(ierr);
    ierr = PetscArraycpy(f, v, Np * dim);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(pvel, &v);CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, ParticleTracerVelocity, &pvel);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode advectParticles(TS ts) {
    TS sts;
    DM sdm;
    Vec particlePosition;
    ParticleData particles;
    PetscReal time;
    PetscErrorCode ierr;
    PetscInt numberLocal; // current number of local particles
    PetscInt numberGlobal; // current number of local particles

    PetscFunctionBeginUser;
    ierr = PetscObjectQuery((PetscObject)ts, "_SwarmTS", (PetscObject *)&sts);CHKERRQ(ierr);
    ierr = TSGetDM(sts, &sdm);CHKERRQ(ierr);
    ierr = TSGetRHSFunction(sts, NULL, NULL, (void **)&particles);CHKERRQ(ierr);

    // if the dm has changed size (new particles, particles moved between ranks, particles deleted) reset the ts
    if (particles->dmChanged){
        ierr = TSReset(sts);CHKERRQ(ierr);
        particles->dmChanged = PETSC_FALSE;
    }

    // Get the current size
    ierr = DMSwarmGetLocalSize(sdm,&numberLocal);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(sdm,&numberGlobal);CHKERRQ(ierr);

    // Get the position vector
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,DMSwarmPICField_coor, &particlePosition);CHKERRQ(ierr);

    // Set the start time for TSSolve
    ierr = TSSetTime(sts, particles->timeInitial);CHKERRQ(ierr);

    // Set the max end time based upon the flow end time
    ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
    ierr = TSSetMaxTime(sts, time);CHKERRQ(ierr);
    particles->timeFinal = time;

    // take the needed timesteps to get to the flow time
    ierr = TSSolve(sts, particlePosition);CHKERRQ(ierr);
    ierr = VecCopy(particles->flowFinal, particles->flowInitial);CHKERRQ(ierr);
    particles->timeInitial = particles->timeFinal;

    // Return the coord vector
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,DMSwarmPICField_coor, &particlePosition);CHKERRQ(ierr);

    // Migrate any particles that have moved
    ierr = DMSwarmMigrate(sdm, PETSC_TRUE);CHKERRQ(ierr);

    // get the new sizes
    PetscInt newNumberLocal;
    PetscInt newNumberGlobal;

    // Get the updated size
    ierr = DMSwarmGetLocalSize(sdm,&newNumberLocal);CHKERRQ(ierr);
    ierr = DMSwarmGetSize(sdm,&newNumberGlobal);CHKERRQ(ierr);

    // Check to see if any of the ranks changed size after migration
    PetscInt dmChanged = newNumberGlobal != numberGlobal ||  newNumberLocal != numberLocal;
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)sts, &comm);CHKERRQ(ierr);
    PetscInt dmChangedAll;
    MPIU_Allreduce(&dmChanged,&dmChangedAll,1,MPIU_INT, MPIU_MAX, comm);
    particles->dmChanged = (PetscBool)dmChangedAll;

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
    ierr = TSSetMaxSteps(particleTs, INT_MAX);CHKERRQ(ierr); // set the max ts to a very large number. This can be over written using ts_max_steps options

    // link the solution with the flowTS
    ierr = TSSetPostStep(flowTs, advectParticles);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)flowTs, "_SwarmTS", (PetscObject)particleTs);CHKERRQ(ierr);
    ierr = DMSwarmVectorDefineField(particles->dm, DMSwarmPICField_coor);CHKERRQ(ierr);

    // Set up the TS
    ierr = TSSetFromOptions(particleTs);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ParticleTracerCreate(ParticleData *particles, PetscInt ndims) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Call the base particle create
    ierr = ParticleCreate(particles, ndims);CHKERRQ(ierr);

    // register all particle fields
    ierr = DMSwarmSetType((*particles)->dm, DMSWARM_PIC);CHKERRQ(ierr);
    ierr = ParticleRegisterPetscDatatypeField(*particles, ParticleTracerVelocity, ndims, PETSC_REAL);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ParticleTracerDestroy(ParticleData *particles) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr = VecDestroy(&((*particles)->flowInitial));CHKERRQ(ierr);

    // Call the base destroy
    ierr = ParticleDestroy(particles);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

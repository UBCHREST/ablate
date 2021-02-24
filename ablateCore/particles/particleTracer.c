#include "particleTracer.h"

/* x_t = v

   Note that here we use the velocity field at t_{n+1} to advect the particles from
   t_n to t_{n+1}. If we use both of these fields, we could use Crank-Nicholson or
   the method of characteristics.
*/
static PetscErrorCode freeStreaming(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
    Particles           particles = (Particles) ctx;
    Vec                 u   = particles->flowInitial;
    DM                  sdm, dm, vdm;
    Vec                 vel, locvel, pvel;
    IS                  vis;
    DMInterpolationInfo ictx;
    const PetscScalar  *coords, *v;
    PetscScalar        *f;
    PetscInt            vf[1] = {0};
    PetscInt            dim, Np;
    PetscErrorCode      ierr;

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
    ierr = DMInterpolationAddPoints(ictx, Np, (PetscReal *) coords);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X, &coords);CHKERRQ(ierr);
    ierr = DMInterpolationSetUp(ictx, vdm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMInterpolationEvaluate(ictx, vdm, locvel, pvel);CHKERRQ(ierr);
    ierr = DMInterpolationDestroy(&ictx);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(vdm, &locvel);CHKERRQ(ierr);
    ierr = DMDestroy(&vdm);CHKERRQ(ierr);

    ierr = VecGetArray(F, &f);CHKERRQ(ierr);
    ierr = VecGetArrayRead(pvel, &v);CHKERRQ(ierr);
    ierr = PetscArraycpy(f, v, Np*dim);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(pvel, &v);CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(sdm, &pvel);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode computeParticleError(TS ts, Vec u, Vec e)
{
    Particles            particles;
    DM                 sdm;
    const PetscScalar *xp0, *xp;
    PetscScalar       *ep;
    PetscReal          time;
    PetscInt           dim, Np, p;
    MPI_Comm           comm;
    PetscErrorCode     ierr;

    PetscFunctionBeginUser;
    ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
    ierr = TSGetApplicationContext(ts, (void **) &particles);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
    ierr = TSGetDM(ts, &sdm);CHKERRQ(ierr);
    ierr = DMGetDimension(sdm, &dim);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(sdm, &Np);CHKERRQ(ierr);
    ierr = VecGetArrayRead(particles->initialLocation, &xp0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u, &xp);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(e, &ep);CHKERRQ(ierr);
    for (p = 0; p < Np; ++p) {
        PetscScalar x[3];
        PetscReal   x0[3];
        PetscInt    d;

        for (d = 0; d < dim; ++d) x0[d] = PetscRealPart(xp0[p*dim+d]);
        ierr = particles->exactSolution(dim, time, x0, 1, x, particles);CHKERRQ(ierr);
        for (d = 0; d < dim; ++d) ep[p*dim+d] += x[d] - xp[p*dim+d];
    }
    ierr = VecRestoreArrayRead(particles->initialLocation, &xp0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u, &xp);CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(e, &ep);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode monitorParticleError(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx)
{
    Particles            adv = (Particles) ctx;
    DM                 sdm;
    const PetscScalar *xp0, *xp;
    PetscReal          error = 0.0;
    PetscInt           dim, Np, p;
    MPI_Comm           comm;
    PetscErrorCode     ierr;

    PetscFunctionBeginUser;
    ierr = PetscObjectGetComm((PetscObject) ts, &comm);CHKERRQ(ierr);
    ierr = TSGetDM(ts, &sdm);CHKERRQ(ierr);
    ierr = DMGetDimension(sdm, &dim);CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(sdm, &Np);CHKERRQ(ierr);
    ierr = VecGetArrayRead(adv->initialLocation, &xp0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(u, &xp);CHKERRQ(ierr);
    for (p = 0; p < Np; ++p) {
        PetscScalar x[3];
        PetscReal   x0[3];
        PetscReal   perror = 0.0;
        PetscInt    d;

        for (d = 0; d < dim; ++d) x0[d] = PetscRealPart(xp0[p*dim+d]);
        ierr = adv->exactSolution(dim, time, x0, 1, x, adv);CHKERRQ(ierr);
        for (d = 0; d < dim; ++d) perror += PetscSqr(PetscRealPart(x[d] - xp[p*dim+d]));
        error += perror;
    }
    ierr = VecRestoreArrayRead(adv->initialLocation, &xp0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u, &xp);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Timestep: %04d time = %-8.4g \t L_2 Particle Error: [%2.3g]\n", (int) step, (double) time, (double) error);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode advectParticles(TS ts)
{
    TS                 sts;
    DM                 sdm;
    Vec                p;
    Particles          particles;
    PetscScalar       *coord, *a;
    const PetscScalar *ca;
    PetscReal          time;
    PetscInt           n;
    PetscErrorCode     ierr;

    PetscFunctionBeginUser;
    ierr = PetscObjectQuery((PetscObject) ts, "_SwarmTS",  (PetscObject *) &sts);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject) ts, "_SwarmSol", (PetscObject *) &p);CHKERRQ(ierr);
    ierr = TSGetDM(sts, &sdm);CHKERRQ(ierr);
    ierr = TSGetRHSFunction(sts, NULL, NULL, (void **) &particles);CHKERRQ(ierr);
    ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coord);CHKERRQ(ierr);
    ierr = VecGetLocalSize(p, &n);CHKERRQ(ierr);
    ierr = VecGetArray(p, &a);CHKERRQ(ierr);
    ierr = PetscArraycpy(a, coord, n);CHKERRQ(ierr);
    ierr = VecRestoreArray(p, &a);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coord);CHKERRQ(ierr);
    ierr = TSGetTime(ts, &time);CHKERRQ(ierr);
    ierr = TSSetMaxTime(sts, time);CHKERRQ(ierr);
    particles->timeFinal = time;
    ierr = TSSolve(sts, p);CHKERRQ(ierr);
    ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coord);CHKERRQ(ierr);
    ierr = VecGetLocalSize(p, &n);CHKERRQ(ierr);
    ierr = VecGetArrayRead(p, &ca);CHKERRQ(ierr);
    ierr = PetscArraycpy(coord, ca, n);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(p, &ca);CHKERRQ(ierr);
    ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coord);CHKERRQ(ierr);

    ierr = VecCopy(particles->flowFinal, particles->flowInitial);CHKERRQ(ierr);
    particles->timeInitial = particles->timeFinal;

    ierr = DMSwarmMigrate(sdm, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMViewFromOptions(sdm, NULL, "-dm_view");CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode getInitialParticleCondition(TS ts, Vec u) {
    Particles particles;
    DM dm;
    PetscFunctionBegin;
    PetscErrorCode ierr = TSGetApplicationContext(ts, &particles);CHKERRQ(ierr);
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

    ierr = VecCopy(particles->initialLocation, u);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ParticleTracerSetupIntegrator(Particles particles, TS particleTs, TS flowTs) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ierr = TSSetDM(particleTs, particles->dm);CHKERRQ(ierr);
    ierr = TSSetProblemType(particleTs, TS_NONLINEAR);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(particleTs, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetApplicationContext(particleTs, particles);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(particleTs, NULL, freeStreaming, particles);CHKERRQ(ierr);

    // link the solution with the flowTS
    ierr = TSSetPostStep(flowTs, advectParticles);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) flowTs, "_SwarmTS", (PetscObject) particleTs);CHKERRQ(ierr);// to else where
    ierr = DMSwarmVectorDefineField(particles->dm, DMSwarmPICField_coor);CHKERRQ(ierr); // do else where
    ierr = DMCreateGlobalVector(particles->dm, &(particles->particleSolution));CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) flowTs, "_SwarmSol", (PetscObject) (particles->particleSolution));CHKERRQ(ierr);// do else where

    // If the exact solution is set, setup the monitors
    if (particles->exactSolution){
        ierr = TSMonitorSet(particleTs, monitorParticleError, particles, NULL);CHKERRQ(ierr);CHKERRQ(ierr);
    }
    ierr = TSSetFromOptions(particleTs);CHKERRQ(ierr);
    if (particles->exactSolution) {
        ierr = TSSetComputeExactError(particleTs, computeParticleError);CHKERRQ(ierr);
    }

    // extract the initial solution
    Vec xtmp;
    ierr = DMCreateGlobalVector(particles->dm, &(particles->initialLocation));CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(particles->dm, DMSwarmPICField_coor, &xtmp);CHKERRQ(ierr);
    ierr = VecCopy(xtmp, particles->initialLocation);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(particles->dm, DMSwarmPICField_coor, &xtmp);CHKERRQ(ierr);

    // setup the initial conditions for error computing
    ierr = TSSetComputeInitialCondition(particleTs, getInitialParticleCondition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ParticleTracerCreate(Particles* particles, PetscInt ndims) {
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

PetscErrorCode ParticleTracerDestroy(Particles* particles) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr = VecDestroy(&((*particles)->flowInitial));CHKERRQ(ierr);
    ierr = VecDestroy(&((*particles)->particleSolution));CHKERRQ(ierr);
    ierr = VecDestroy(&((*particles)->initialLocation));CHKERRQ(ierr);

    // Call the base destroy
    ierr = ParticleDestroy(particles);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

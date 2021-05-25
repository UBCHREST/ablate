#include "particleInertial.h"
#include <petsc/private/dmswarmimpl.h>
#include <petscdmswarm.h>
#include "incompressibleFlow.h"

const char FluidVelocity[] = "FluidVelocity";
const char ParticleKinematics[] = "ParticleKinematics";
enum InertialParticleFields {Position, Velocity, TotalParticleField};

/*
 * Kinematics vector is a combination of particle velocity and position
 * in order to pass into TSSolve.
*/
static PetscErrorCode PackKinematics(TS ts, Vec position, Vec velocity, Vec kinematics){

    const PetscScalar *pos, *vel;
    PetscScalar *kin;
    PetscInt           Np,p,dim,n;
    PetscErrorCode     ierr;
    DM dm;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = VecGetArray(kinematics, &kin);CHKERRQ(ierr);
    ierr = VecGetArrayRead(position, &pos);CHKERRQ(ierr);
    ierr = VecGetArrayRead(velocity, &vel);CHKERRQ(ierr);
    ierr = VecGetLocalSize(position, &Np);CHKERRQ(ierr);

    Np /= dim;
    for (p = 0; p < Np; ++p) {
        for (n = 0; n < dim; n++) {
            kin[p*TotalParticleField*dim+n] = pos[ p * dim + n];
            kin[p*TotalParticleField*dim+dim+n] = vel[p * dim + n];
        }
    }
    ierr = VecRestoreArray(kinematics, &kin);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(position, &pos);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(velocity, &vel);CHKERRQ(ierr);
    PetscFunctionReturn(0);
};

/*
 * Unpack kinematics to get particle position and velocity
 */
static PetscErrorCode UnpackKinematics(TS ts, Vec kinematics, Vec position, Vec velocity){

    PetscScalar *pos, *vel;
    const PetscScalar *kin;
    PetscInt           Np,p,dim,n;
    PetscErrorCode     ierr;
    DM dm;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = VecGetArrayRead(kinematics, &kin);CHKERRQ(ierr);
    ierr = VecGetArray(position, &pos);CHKERRQ(ierr);
    ierr = VecGetArray(velocity, &vel);CHKERRQ(ierr);
    ierr = VecGetLocalSize(position, &Np);CHKERRQ(ierr);

    Np /= dim;
    for (p = 0; p < Np; ++p) {
        for (n = 0; n < dim; n++) {
            pos[p * dim +n] = kin[p*TotalParticleField*dim+n];
            vel[p * dim +n] = kin[p*TotalParticleField*dim+dim+n];
        }
    }
    ierr = VecRestoreArrayRead(kinematics, &kin);CHKERRQ(ierr);
    ierr = VecRestoreArray(position, &pos);CHKERRQ(ierr);
    ierr = VecRestoreArray(velocity, &vel);CHKERRQ(ierr);
    PetscFunctionReturn(0);
};

/* calculating RHS of the following equations
 * x_t = vp
 * u_t = f(vf-vp)/tau_p + g(1-\rho_f/\rho_p)

   Note that here we use the velocity field at t_{n+1} to advect the particles from
   t_n to t_{n+1}. If we use both of these fields, we could use Crank-Nicholson or
   the method of characteristics.
*/
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {

    ParticleData particles = (ParticleData)ctx;
    InertialParticleParameters * partParameters = (InertialParticleParameters *)particles->data;
    Vec u = particles->flowFinal;
    DM sdm, dm, vdm;
    Vec vel, locvel, fluidVelocity;
    IS vis;
    DMInterpolationInfo ictx;
    const PetscScalar *coords;
    PetscScalar *f;
    PetscInt vf[1] = {particles->flowVelocityFieldIndex};
    PetscInt dim, Np;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &sdm);CHKERRQ(ierr);
    ierr = DMSwarmGetCellDM(sdm, &dm);CHKERRQ(ierr);
    // fluidVelocity is at particle location.
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, FluidVelocity, &fluidVelocity);CHKERRQ(ierr);

    ierr = DMSwarmGetLocalSize(sdm, &Np);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    /* Get local fluid velocity */
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

    //Get particle position, velocity, diameter and density
    Vec particlePosition, particleVelocity, particleDiameter, particleDensity;
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,DMSwarmPICField_coor, &particlePosition);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,ParticleVelocity, &particleVelocity);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,ParticleDiameter, &particleDiameter);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,ParticleDensity, &particleDensity);CHKERRQ(ierr);

    //Unpack kinematics to get updated position and velocity
    ierr = UnpackKinematics(ts, X, particlePosition, particleVelocity);CHKERRQ(ierr);
    ierr = VecGetArrayRead(particlePosition, &coords);CHKERRQ(ierr);
    ierr = DMInterpolationAddPoints(ictx, Np, (PetscReal *)coords);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(particlePosition, &coords);CHKERRQ(ierr);

    /* Particles that lie outside the domain should be dropped,
     whereas particles that move to another partition should trigger a migration */
    ierr = DMInterpolationSetUp(ictx, vdm, PETSC_FALSE, PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecSet(fluidVelocity, 0.);CHKERRQ(ierr);

    ierr = DMInterpolationEvaluate(ictx, vdm, locvel, fluidVelocity);CHKERRQ(ierr);
    ierr = DMInterpolationDestroy(&ictx);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(vdm, &locvel);CHKERRQ(ierr);
    ierr = DMDestroy(&vdm);CHKERRQ(ierr);

    //Calculate RHS of particle position and velocity equations
    PetscInt p,n;
    const PetscScalar *partVel, *fluidVel, *partDiam, *partDens;
    PetscReal g[3] = {partParameters->gravityField[0], partParameters->gravityField[1],partParameters->gravityField[2]}; // gravity field
    PetscScalar muF = partParameters->fluidViscosity;
    PetscScalar rhoF = partParameters->fluidDensity;

    PetscReal Rep, corFactor;
    PetscScalar tauP; // particle Stokes relaxation time

    ierr = VecGetArray(F, &f);CHKERRQ(ierr);
    ierr = VecGetArrayRead(particleVelocity, &partVel);CHKERRQ(ierr);
    ierr = VecGetArrayRead(fluidVelocity, &fluidVel);CHKERRQ(ierr);
    ierr = VecGetArrayRead(particleDensity, &partDens);CHKERRQ(ierr);
    ierr = VecGetArrayRead(particleDiameter, &partDiam);CHKERRQ(ierr);

    for (p = 0; p < Np; ++p) {
        Rep = 0.0;
        for (n = 0; n < dim; n++){
            Rep += rhoF*PetscSqr(fluidVel[p * dim + n]- partVel[p * dim + n])*partDiam[p]/muF;
        }
        //Correction factor to account for finite Rep on Stokes drag (see Schiller-Naumann drag closure)
        corFactor = 1.0+0.15*PetscPowReal(PetscSqrtReal(Rep),0.687);
        if (Rep < 0.1) {
            corFactor =1.0; //returns Stokes drag for low speed particles
        }
        tauP = partDens[p]*PetscSqr(partDiam[p])/(18.0*muF); // particle relaxation time
        for (n = 0; n < dim; n++) {
            f[p*TotalParticleField*dim+n] = partVel[p * dim + n];
            f[p*TotalParticleField*dim+dim+n] = corFactor*(fluidVel[p * dim + n]- partVel[p *dim + n])/tauP + g[n]*(1.0-rhoF/partDens[p]);
        }
    }
    ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(particleVelocity, &partVel);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(fluidVelocity, &fluidVel);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(particleDensity, &partDens);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(particleDiameter, &partDiam);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, FluidVelocity, &fluidVelocity);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,DMSwarmPICField_coor, &particlePosition);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,ParticleVelocity, &particleVelocity);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,ParticleDiameter, &particleDiameter);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,ParticleDensity, &particleDensity);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode advectParticles(TS flowTS, void* ctx) {
    TS sts;
    DM sdm;
    Vec particlePosition, particleVelocity, particleKinematics;
    ParticleData particles;
    PetscReal time;
    PetscErrorCode ierr;
    PetscInt numberLocal; // current number of local particles
    PetscInt numberGlobal; // current number of local particles

    PetscFunctionBeginUser;
    sts = (TS)ctx;
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

    // Get the position, velocity and Kinematics vector
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,DMSwarmPICField_coor, &particlePosition);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,ParticleVelocity, &particleVelocity);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,ParticleKinematics, &particleKinematics);CHKERRQ(ierr);

    // combine the position and velocity vectors in kinematics as a unit vector to solve by TSSolve
    ierr = PackKinematics(sts, particlePosition, particleVelocity, particleKinematics);CHKERRQ(ierr);

    // get the particle time step
    PetscReal dtInitial;
    ierr = TSGetTimeStep(sts, &dtInitial);CHKERRQ(ierr);

    // Set the max end time based upon the flow end time
    ierr = TSGetTime(flowTS, &time);CHKERRQ(ierr);
    ierr = TSSetMaxTime(sts, time);CHKERRQ(ierr);
    particles->timeFinal = time;

    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,DMSwarmPICField_coor, &particlePosition);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,ParticleVelocity, &particleVelocity);CHKERRQ(ierr);

    // take the needed timesteps to get to the flow time
    ierr = TSSolve(sts, particleKinematics);CHKERRQ(ierr);

    ierr = VecCopy(particles->flowFinal, particles->flowInitial);CHKERRQ(ierr);
    particles->timeInitial = particles->timeFinal;

    // get the updated time step, and reset if it has gone down
    PetscReal dtUpdated;
    ierr = TSGetTimeStep(sts, &dtUpdated);CHKERRQ(ierr);
    if (dtUpdated < dtInitial){
        ierr = TSSetTimeStep(sts, dtInitial);CHKERRQ(ierr);
    }

    //get position and velocity vectors
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,DMSwarmPICField_coor, &particlePosition);CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm,ParticleVelocity, &particleVelocity);CHKERRQ(ierr);

    // unpack kinematics to get particle position and velocity separately
    ierr = UnpackKinematics(sts, particleKinematics, particlePosition, particleVelocity);CHKERRQ(ierr);

    // Return position, velocity and kinematics vectors
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,ParticleKinematics, &particleKinematics);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,DMSwarmPICField_coor, &particlePosition);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm,ParticleVelocity, &particleVelocity);CHKERRQ(ierr);

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

PetscErrorCode ParticleInertialSetupIntegrator(ParticleData particles, TS particleTs, FlowData flowData) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ierr = TSSetDM(particleTs, particles->dm);CHKERRQ(ierr);
    ierr = TSSetProblemType(particleTs, TS_NONLINEAR);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(particleTs, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetApplicationContext(particleTs, particles);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(particleTs, NULL, RHSFunction, particles);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(particleTs, 100000000);CHKERRQ(ierr); // set the max ts to a very large number. This can be over written using ts_max_steps options

    // Set the start time for TSSolve
    ierr = TSSetTime(particleTs, particles->timeInitial);CHKERRQ(ierr);

    // link the solution with the flowTS
    ierr = FlowRegisterPostStep(flowData, advectParticles, particleTs);CHKERRQ(ierr);

    // Set up the TS
    ierr = TSSetFromOptions(particleTs);CHKERRQ(ierr);
    ierr = TSViewFromOptions(particleTs,NULL, "-ts_view");CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ParticleInertialCreate(ParticleData *particles, PetscInt ndims) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Call the base particle create
    ierr = ParticleCreate(particles, ndims);CHKERRQ(ierr);

    // register all particle fields
    ierr = DMSwarmSetType((*particles)->dm, DMSWARM_PIC);CHKERRQ(ierr);
    ierr = ParticleRegisterPetscDatatypeField(*particles, FluidVelocity, ndims, PETSC_REAL);CHKERRQ(ierr);
    ierr = ParticleRegisterPetscDatatypeField(*particles, ParticleVelocity, ndims, PETSC_REAL);CHKERRQ(ierr);
    ierr = ParticleRegisterPetscDatatypeField(*particles, ParticleKinematics, TotalParticleField*ndims, PETSC_REAL);CHKERRQ(ierr);
    ierr = ParticleRegisterPetscDatatypeField(*particles, ParticleDiameter, 1, PETSC_REAL);CHKERRQ(ierr);
    ierr = ParticleRegisterPetscDatatypeField(*particles, ParticleDensity, 1, PETSC_REAL);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ParticleInertialDestroy(ParticleData *particles) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr = VecDestroy(&((*particles)->flowInitial));CHKERRQ(ierr);

    // Call the base destroy
    ierr = ParticleDestroy(particles);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

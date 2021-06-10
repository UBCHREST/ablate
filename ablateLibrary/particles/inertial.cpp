#include "inertial.hpp"
#include <utilities/petscError.hpp>

enum InertialParticleFields { Position, Velocity, TotalParticleField };

ablate::particles::Inertial::Inertial(std::string name, int ndims, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<particles::initializers::Initializer> initializer,
                                      std::vector<std::shared_ptr<mathFunctions::FieldSolution>> fieldInitialization, std::shared_ptr<mathFunctions::MathFunction> exactSolution, std::shared_ptr<parameters::Parameters> options)
    : Particles(name, ndims, initializer, fieldInitialization, exactSolution, options) {
    RegisterField(ParticleFieldDescriptor{.fieldName = FluidVelocity, .components = ndims, .type = PETSC_REAL});
    RegisterField(ParticleFieldDescriptor{.fieldName = ParticleVelocity, .components = ndims, .type = PETSC_REAL});
    RegisterField(ParticleFieldDescriptor{.fieldName = ParticleKinematics, .components = TotalParticleField * ndims, .type = PETSC_REAL});
    RegisterField(ParticleFieldDescriptor{.fieldName = ParticleDiameter, .components = 1, .type = PETSC_REAL});
    RegisterField(ParticleFieldDescriptor{.fieldName = ParticleDensity, .components = 1, .type = PETSC_REAL});

    // initialize the constant values
    fluidDensity = parameters->GetExpect<PetscReal>("fluidDensity");
    fluidViscosity = parameters->GetExpect<PetscReal>("fluidViscosity");
    auto gravityVector = parameters->GetExpect<std::vector<PetscReal>>("gravityField");
    for(auto i = 0; i < PetscMin(gravityVector.size(), 3); i++){
        gravityField[i] = gravityVector[i];
    }
}
ablate::particles::Inertial::~Inertial() {}
void ablate::particles::Inertial::InitializeFlow(std::shared_ptr<flow::Flow> flow) {
    // Call the base to initialize the flow
    Particles::InitializeFlow(flow);

    TSSetRHSFunction(particleTs, NULL, RHSFunction, this) >> checkError;

    // Set the start time for TSSolve
    TSSetTime(particleTs, timeInitial) >> checkError;

    // link the solution with the flowTS
    flow->RegisterPostStep([this](TS flowTs, ablate::flow::Flow&){
      this->advectParticles(flowTs);
    });
}

PetscErrorCode ablate::particles::Inertial::PackKinematics(TS ts, Vec position, Vec velocity, Vec kinematics) {
    const PetscScalar *pos, *vel;
    PetscScalar *kin;
    PetscInt Np, p, dim, n;
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);
    ierr = VecGetArray(kinematics, &kin);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(position, &pos);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(velocity, &vel);
    CHKERRQ(ierr);
    ierr = VecGetLocalSize(position, &Np);
    CHKERRQ(ierr);

    Np /= dim;
    for (p = 0; p < Np; ++p) {
        for (n = 0; n < dim; n++) {
            kin[p * TotalParticleField * dim + n] = pos[p * dim + n];
            kin[p * TotalParticleField * dim + dim + n] = vel[p * dim + n];
        }
    }
    ierr = VecRestoreArray(kinematics, &kin);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(position, &pos);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(velocity, &vel);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::particles::Inertial::UnpackKinematics(TS ts, Vec kinematics, Vec position, Vec velocity) {
    PetscScalar *pos, *vel;
    const PetscScalar *kin;
    PetscInt Np, p, dim, n;
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(kinematics, &kin);
    CHKERRQ(ierr);
    ierr = VecGetArray(position, &pos);
    CHKERRQ(ierr);
    ierr = VecGetArray(velocity, &vel);
    CHKERRQ(ierr);
    ierr = VecGetLocalSize(position, &Np);
    CHKERRQ(ierr);

    Np /= dim;
    for (p = 0; p < Np; ++p) {
        for (n = 0; n < dim; n++) {
            pos[p * dim + n] = kin[p * TotalParticleField * dim + n];
            vel[p * dim + n] = kin[p * TotalParticleField * dim + dim + n];
        }
    }
    ierr = VecRestoreArrayRead(kinematics, &kin);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(position, &pos);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(velocity, &vel);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::particles::Inertial::RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
    PetscFunctionBeginUser;

    ablate::particles::Inertial* particles = (ablate::particles::Inertial*)ctx;

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
    ierr = TSGetDM(ts, &sdm);
    CHKERRQ(ierr);
    ierr = DMSwarmGetCellDM(sdm, &dm);
    CHKERRQ(ierr);
    // fluidVelocity is at particle location.
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, FluidVelocity, &fluidVelocity);
    CHKERRQ(ierr);

    ierr = DMSwarmGetLocalSize(sdm, &Np);
    CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);

    /* Get local fluid velocity */
    ierr = DMCreateSubDM(dm, 1, vf, &vis, &vdm);
    CHKERRQ(ierr);
    ierr = VecGetSubVector(u, vis, &vel);
    CHKERRQ(ierr);
    ierr = DMGetLocalVector(vdm, &locvel);
    CHKERRQ(ierr);
    ierr = DMPlexInsertBoundaryValues(vdm, PETSC_TRUE, locvel, particles->timeInitial, NULL, NULL, NULL);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(vdm, vel, INSERT_VALUES, locvel);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(vdm, vel, INSERT_VALUES, locvel);
    CHKERRQ(ierr);
    ierr = VecRestoreSubVector(u, vis, &vel);
    CHKERRQ(ierr);
    ierr = ISDestroy(&vis);
    CHKERRQ(ierr);

    /* Interpolate velocity */
    ierr = DMInterpolationCreate(PETSC_COMM_SELF, &ictx);
    CHKERRQ(ierr);
    ierr = DMInterpolationSetDim(ictx, dim);
    CHKERRQ(ierr);
    ierr = DMInterpolationSetDof(ictx, dim);
    CHKERRQ(ierr);

    // Get particle position, velocity, diameter and density
    Vec particlePosition, particleVelocity, particleDiameter, particleDensity;
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, DMSwarmPICField_coor, &particlePosition);
    CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, ParticleVelocity, &particleVelocity);
    CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, ParticleDiameter, &particleDiameter);
    CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, ParticleDensity, &particleDensity);
    CHKERRQ(ierr);

    // Unpack kinematics to get updated position and velocity
    ierr = UnpackKinematics(ts, X, particlePosition, particleVelocity);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(particlePosition, &coords);
    CHKERRQ(ierr);
    ierr = DMInterpolationAddPoints(ictx, Np, (PetscReal *)coords);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(particlePosition, &coords);
    CHKERRQ(ierr);

    /* Particles that lie outside the domain should be dropped,
     whereas particles that move to another partition should trigger a migration */
    ierr = DMInterpolationSetUp(ictx, vdm, PETSC_FALSE, PETSC_TRUE);
    CHKERRQ(ierr);
    ierr = VecSet(fluidVelocity, 0.);
    CHKERRQ(ierr);

    ierr = DMInterpolationEvaluate(ictx, vdm, locvel, fluidVelocity);
    CHKERRQ(ierr);
    ierr = DMInterpolationDestroy(&ictx);
    CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(vdm, &locvel);
    CHKERRQ(ierr);
    ierr = DMDestroy(&vdm);
    CHKERRQ(ierr);

    // Calculate RHS of particle position and velocity equations
    PetscInt p, n;
    const PetscScalar *partVel, *fluidVel, *partDiam, *partDens;
    PetscReal g[3] = {particles->gravityField[0], particles->gravityField[1], particles->gravityField[2]};  // gravity field
    PetscScalar muF = particles->fluidViscosity;
    PetscScalar rhoF = particles->fluidDensity;

    PetscReal Rep, corFactor;
    PetscScalar tauP;  // particle Stokes relaxation time

    ierr = VecGetArray(F, &f);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(particleVelocity, &partVel);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(fluidVelocity, &fluidVel);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(particleDensity, &partDens);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(particleDiameter, &partDiam);
    CHKERRQ(ierr);

    for (p = 0; p < Np; ++p) {
        Rep = 0.0;
        for (n = 0; n < dim; n++) {
            Rep += rhoF * PetscSqr(fluidVel[p * dim + n] - partVel[p * dim + n]) * partDiam[p] / muF;
        }
        // Correction factor to account for finite Rep on Stokes drag (see Schiller-Naumann drag closure)
        corFactor = 1.0 + 0.15 * PetscPowReal(PetscSqrtReal(Rep), 0.687);
        if (Rep < 0.1) {
            corFactor = 1.0;  // returns Stokes drag for low speed particles
        }
        tauP = partDens[p] * PetscSqr(partDiam[p]) / (18.0 * muF);  // particle relaxation time
        for (n = 0; n < dim; n++) {
            f[p * TotalParticleField * dim + n] = partVel[p * dim + n];
            f[p * TotalParticleField * dim + dim + n] = corFactor * (fluidVel[p * dim + n] - partVel[p * dim + n]) / tauP + g[n] * (1.0 - rhoF / partDens[p]);
        }
    }
    ierr = VecRestoreArray(F, &f);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(particleVelocity, &partVel);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(fluidVelocity, &fluidVel);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(particleDensity, &partDens);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(particleDiameter, &partDiam);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, FluidVelocity, &fluidVelocity);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, DMSwarmPICField_coor, &particlePosition);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, ParticleVelocity, &particleVelocity);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, ParticleDiameter, &particleDiameter);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, ParticleDensity, &particleDensity);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
void ablate::particles::Inertial::advectParticles(TS flowTS) {

    Vec particlePosition, particleVelocity, particleKinematics;
    PetscReal time;

    // if the dm has changed size (new particles, particles moved between ranks, particles deleted) reset the ts
    if (dmChanged){
        TSReset(particleTs) >> checkError;
        dmChanged = PETSC_FALSE;
    }

    // Get the position, velocity and Kinematics vector
    DMSwarmCreateGlobalVectorFromField(dm, DMSwarmPICField_coor, &particlePosition) >> checkError;
    DMSwarmCreateGlobalVectorFromField(dm, ParticleVelocity, &particleVelocity) >> checkError;
    DMSwarmCreateGlobalVectorFromField(dm, ParticleKinematics, &particleKinematics) >> checkError;

    // combine the position and velocity vectors in kinematics as a unit vector to solve by TSSolve
    PackKinematics(particleTs, particlePosition, particleVelocity, particleKinematics) >> checkError;

    // get the particle time step
    PetscReal dtInitial;
    TSGetTimeStep(particleTs, &dtInitial) >> checkError;

    // Set the max end time based upon the flow end time
    TSGetTime(flowTS, &time) >> checkError;
    TSSetMaxTime(particleTs, time) >> checkError;
    timeFinal = time;

    DMSwarmDestroyGlobalVectorFromField(dm, DMSwarmPICField_coor, &particlePosition) >> checkError;
    DMSwarmDestroyGlobalVectorFromField(dm, ParticleVelocity, &particleVelocity) >> checkError;

    // take the needed timesteps to get to the flow time
    TSSolve(particleTs, particleKinematics) >> checkError;

    VecCopy(flowFinal, flowInitial) >> checkError;
    timeInitial = timeFinal;

    // get the updated time step, and reset if it has gone down
    PetscReal dtUpdated;
    TSGetTimeStep(particleTs, &dtUpdated) >> checkError;
    if (dtUpdated < dtInitial) {
        TSSetTimeStep(particleTs, dtInitial) >> checkError;
    }

    // get position and velocity vectors
    DMSwarmCreateGlobalVectorFromField(dm, DMSwarmPICField_coor, &particlePosition) >> checkError;
    DMSwarmCreateGlobalVectorFromField(dm, ParticleVelocity, &particleVelocity) >> checkError;

    // unpack kinematics to get particle position and velocity separately
    UnpackKinematics(particleTs, particleKinematics, particlePosition, particleVelocity) >> checkError;

    // Return position, velocity and kinematics vectors
    DMSwarmDestroyGlobalVectorFromField(dm, ParticleKinematics, &particleKinematics) >> checkError;
    DMSwarmDestroyGlobalVectorFromField(dm, DMSwarmPICField_coor, &particlePosition) >> checkError;
    DMSwarmDestroyGlobalVectorFromField(dm, ParticleVelocity, &particleVelocity) >> checkError;

    // Migrate any particles that have moved
    Particles::SwarmMigrate();
}

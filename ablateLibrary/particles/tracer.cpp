#include "tracer.hpp"
#include "parser/registrar.hpp"
#include "solve/timeStepper.hpp"
#include "utilities/petscError.hpp"

ablate::particles::Tracer::Tracer(std::string name, int ndims, std::shared_ptr<particles::initializers::Initializer> initializer, std::shared_ptr<mathFunctions::MathFunction> exactSolution,
                                  std::shared_ptr<parameters::Parameters> options)
    : Particles(name, ndims, initializer, exactSolution, options) {
    RegisterField(ParticleFieldDescriptor{.fieldName = ParticleVelocity, .components = ndims, .type = PETSC_REAL});
}

ablate::particles::Tracer::~Tracer() {}

void ablate::particles::Tracer::InitializeFlow(std::shared_ptr<flow::Flow> flow) {
    // Call the base to initialize the flow
    Particles::InitializeFlow(flow);

    TSSetRHSFunction(particleTs, NULL, freeStreaming, this) >> checkError;

    // Set the start time for TSSolve
    TSSetTime(particleTs, timeInitial) >> checkError;

    // link the solution with the flowTS
    flow->RegisterPostStep([this](TS flowTs, ablate::flow::Flow&){
        this->advectParticles(flowTs);
    });
}

/* x_t = v

   Note that here we use the velocity field at t_{n+1} to advect the particles from
   t_n to t_{n+1}. If we use both of these fields, we could use Crank-Nicholson or
   the method of characteristics.
*/
PetscErrorCode ablate::particles::Tracer::freeStreaming(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
    ablate::particles::Tracer* particles = (ablate::particles::Tracer*)ctx;
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
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, ParticleVelocity, &pvel);CHKERRQ(ierr);
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

    // right hand side storing
    ierr = VecGetArray(F, &f);CHKERRQ(ierr);
    ierr = VecGetArrayRead(pvel, &v);CHKERRQ(ierr);
    ierr = PetscArraycpy(f, v, Np * dim);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(pvel, &v);CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, ParticleVelocity, &pvel);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

void ablate::particles::Tracer::advectParticles(TS flowTS) {
    Vec particlePosition;
    PetscReal time;

    // if the dm has changed size (new particles, particles moved between ranks, particles deleted) reset the ts
    if (dmChanged){
        TSReset(particleTs) >> checkError;
        dmChanged = PETSC_FALSE;
    }

    // Get the position vector
    DMSwarmCreateGlobalVectorFromField(dm,DMSwarmPICField_coor, &particlePosition) >> checkError;

    // get the particle time step
    PetscReal dtInitial;
    TSGetTimeStep(particleTs, &dtInitial) >> checkError;

    // Set the max end time based upon the flow end time
    TSGetTime(flowTS, &time) >> checkError;
    TSSetMaxTime(particleTs, time) >> checkError;
    timeFinal = time;

    // take the needed timesteps to get to the flow time
    TSSolve(particleTs, particlePosition) >> checkError;
    VecCopy(flowFinal, flowInitial) >> checkError;
    timeInitial = timeFinal;

    // get the updated time step, and reset if it has gone down
    PetscReal dtUpdated;
    TSGetTimeStep(particleTs, &dtUpdated) >> checkError;
    if (dtUpdated < dtInitial){
        TSSetTimeStep(particleTs, dtInitial) >> checkError;
    }

    // Return the coord vector
    DMSwarmDestroyGlobalVectorFromField(dm,DMSwarmPICField_coor, &particlePosition) >> checkError;

    // Migrate any particles that have moved
    Particles::SwarmMigrate();
}

// REGISTER(ablate::particles::Particles, ablate::particles::Tracer, "massless particles that advect with the flow", ARG(std::string, "name", "the name of the particle group"),
//          ARG(int, "ndims", "the number of dimensions for the particle"), ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"),
//          ARG(particles::initializers::Initializer, "initializer", "the initial particle setup methods"), ARG(mathFunctions::MathFunction, "exactSolution", "the particle location exact solution"));
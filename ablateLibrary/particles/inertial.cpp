#include "inertial.hpp"
#include <utilities/petscError.hpp>

enum InertialParticleFields { Position, Velocity, TotalParticleField };

ablate::particles::Inertial::Inertial(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, int ndims,
                                      std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<particles::initializers::Initializer> initializer,
                                      std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization, std::shared_ptr<mathFunctions::MathFunction> exactSolution)
    : Particles(solverId, region, options, ndims,
                {
                    ParticleField{.name = ParticleVelocity, .components = CreateDimensionVector("VEL_", ndims), .type = domain::FieldType::SOL, .dataType = PETSC_REAL},
                    ParticleField{.name = FluidVelocity, .components = CreateDimensionVector("FLUID_VEL_", ndims), .type = domain::FieldType::AUX, .dataType = PETSC_REAL},
                    ParticleField{.name = ParticleDiameter, .type = domain::FieldType::AUX, .dataType = PETSC_REAL},
                    ParticleField{.name = ParticleDensity, .type = domain::FieldType::AUX, .dataType = PETSC_REAL},
                },
                initializer, fieldInitialization, exactSolution) {
    // initialize the constant values
    fluidDensity = parameters->GetExpect<PetscReal>("fluidDensity");
    fluidViscosity = parameters->GetExpect<PetscReal>("fluidViscosity");
    auto gravityVector = parameters->GetExpect<std::vector<PetscReal>>("gravityField");
    for (std::size_t i = 0; i < PetscMin(gravityVector.size(), 3); i++) {
        gravityField[i] = gravityVector[i];
    }
}
ablate::particles::Inertial::~Inertial() {}
void ablate::particles::Inertial::Initialize() {
    // Call the base to initialize the flow
    Particles::Initialize();

    TSSetRHSFunction(particleTs, NULL, RHSFunction, this) >> checkError;

    // Set the start time for TSSolve
    TSSetTime(particleTs, timeInitial) >> checkError;

    // link the solution with the flowTS
    RegisterPostStep([this](TS flowTs, ablate::solver::Solver &) { this->AdvectParticles(flowTs); });
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

    ablate::particles::Inertial *particles = (ablate::particles::Inertial *)ctx;

    Vec u = particles->flowFinal;
    DM sdm, dm, vdm;
    Vec vel, locvel, fluidVelocity;
    IS vis;
    DMInterpolationInfo ictx;
    const PetscScalar *coords;
    PetscScalar *f;
    PetscInt vf[1] = {particles->flowVelocityField.id};
    PetscInt dim, Np;
    PetscErrorCode ierr;

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
        // Note: this function assumed that the solution vector order is correct
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

#include "parser/registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::particles::Inertial, "particles (with mass) that advect with the flow", ARG(std::string, "id", "the name of this particle solver"),
         OPT(domain::Region, "region", "the region to apply this solver.  Default is entire domain"), OPT(ablate::parameters::Parameters, "options", "options for the flow passed directly to PETSc"),
         ARG(int, "ndims", "the number of dimensions for the particle"), ARG(parameters::Parameters, "parameters", "fluid parameters for the particles (fluidDensity, fluidViscosity, gravityField)"),
         ARG(particles::initializers::Initializer, "initializer", "the initial particle setup methods"),
         ARG(std::vector<mathFunctions::FieldFunction>, "fieldInitialization", "the initial particle fields setup methods"),
         OPT(mathFunctions::MathFunction, "exactSolution", "the particle location/velocity exact solution"));
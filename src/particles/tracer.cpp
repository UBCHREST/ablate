#include "tracer.hpp"
#include "solver/timeStepper.hpp"
#include "utilities/petscError.hpp"

ablate::particles::Tracer::Tracer(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, int ndims,
                                  std::shared_ptr<particles::initializers::Initializer> initializer, std::shared_ptr<mathFunctions::MathFunction> exactSolution)
    : Particles(solverId, region, options, ndims,
                {ParticleField{.name = ParticleVelocity, .components = CreateDimensionVector("VEL_", ndims), .type = domain::FieldLocation::AUX, .dataType = PETSC_REAL}}, initializer, {},
                exactSolution) {}

ablate::particles::Tracer::~Tracer() {}

void ablate::particles::Tracer::Initialize() {
    // Call the base to initialize the flow
    Particles::Initialize();

    TSSetRHSFunction(particleTs, NULL, freeStreaming, this) >> checkError;

    // Set the start time for TSSolve
    TSSetTime(particleTs, timeInitial) >> checkError;

    // link the solution with the flowTS
    RegisterPostStep([this](TS flowTs, ablate::solver::Solver &) { this->AdvectParticles(flowTs); });
}

/* x_t = v

   Note that here we use the velocity field at t_{n+1} to advect the particles from
   t_n to t_{n+1}. If we use both of these fields, we could use Crank-Nicholson or
   the method of characteristics.
*/
PetscErrorCode ablate::particles::Tracer::freeStreaming(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
    ablate::particles::Tracer *particles = (ablate::particles::Tracer *)ctx;
    //    Vec u = particles->flowFinal;
    DM sdm, dm, vdm;
    Vec locvel, pvel;
    IS vis;
    DMInterpolationInfo ictx;
    const PetscScalar *coords, *v;
    PetscScalar *f;
    PetscInt dim, Np;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &sdm);
    CHKERRQ(ierr);
    ierr = DMSwarmGetCellDM(sdm, &dm);
    CHKERRQ(ierr);
    ierr = DMSwarmCreateGlobalVectorFromField(sdm, ParticleVelocity, &pvel);
    CHKERRQ(ierr);
    ierr = DMSwarmGetLocalSize(sdm, &Np);
    CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);

    /* Get local flow velocity */
    const auto &flowVelocityField = particles->subDomain->GetField("velocity");
    ierr = particles->subDomain->GetFieldLocalVector(flowVelocityField, particles->timeInitial, &vis, &locvel, &vdm);
    CHKERRQ(ierr);

    /* Interpolate velocity */
    ierr = DMInterpolationCreate(PETSC_COMM_SELF, &ictx);
    CHKERRQ(ierr);
    ierr = DMInterpolationSetDim(ictx, dim);
    CHKERRQ(ierr);
    ierr = DMInterpolationSetDof(ictx, dim);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(X, &coords);
    CHKERRQ(ierr);
    ierr = DMInterpolationAddPoints(ictx, Np, (PetscReal *)coords);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X, &coords);
    CHKERRQ(ierr);

    /* Particles that lie outside the domain should be dropped,
     whereas particles that move to another partition should trigger a migration */
    ierr = DMInterpolationSetUp(ictx, vdm, PETSC_FALSE, PETSC_TRUE);
    CHKERRQ(ierr);
    ierr = VecSet(pvel, 0.);
    CHKERRQ(ierr);

    ierr = DMInterpolationEvaluate(ictx, vdm, locvel, pvel);
    CHKERRQ(ierr);
    ierr = DMInterpolationDestroy(&ictx);
    CHKERRQ(ierr);
    ierr = particles->subDomain->RestoreFieldLocalVector(flowVelocityField, &vis, &locvel, &vdm);
    CHKERRQ(ierr);

    // right hand side storing
    ierr = VecGetArray(F, &f);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(pvel, &v);
    CHKERRQ(ierr);
    ierr = PetscArraycpy(f, v, Np * dim);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(pvel, &v);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &f);
    CHKERRQ(ierr);
    ierr = DMSwarmDestroyGlobalVectorFromField(sdm, ParticleVelocity, &pvel);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::particles::Tracer, "massless particles that advect with the flow", ARG(std::string, "id", "the name of this particle solver"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "options for the flow passed directly to PETSc"), ARG(int, "ndims", "the number of dimensions for the particle"),
         ARG(ablate::particles::initializers::Initializer, "initializer", "the initial particle setup methods"),
         OPT(ablate::mathFunctions::MathFunction, "exactSolution", "the particle location exact solution"));
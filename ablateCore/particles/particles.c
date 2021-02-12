#include "incompressibleFlow.h"
#include "particles.h"
#include "particleTracer.h"

PetscErrorCode ParticleCreate(Particles* particles,ParticleType type, Flow flow, ParticleInitializer particleInitializer) {
    PetscErrorCode ierr;
    *particles = malloc(sizeof(struct _Particles));

    // initialize all fields
    (*particles)->flow = flow;
    (*particles)->dm = NULL;
    (*particles)->parameters = NULL;
    (*particles)->data = NULL;
    (*particles)->setupIntegrator = NULL;
    (*particles)->destroy = NULL;
    (*particles)->exactSolution = NULL;
    (*particles)->particleInitializer = particleInitializer;

    // create and associate the dm
    ierr = DMCreate(PETSC_COMM_WORLD, &(*particles)->dm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)((*particles)->dm), "particles_");CHKERRQ(ierr);
    ierr = DMSetType((*particles)->dm, DMSWARM);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) (*particles)->dm, "Particles");CHKERRQ(ierr);

    // get the dimensions of the flow and set for the particle dm
    PetscInt dim;
    ierr = DMGetDimension(flow->dm, &dim);CHKERRQ(ierr);
    ierr = DMSetDimension((*particles)->dm, dim);CHKERRQ(ierr);
    ierr = DMSwarmSetCellDM((*particles)->dm, flow->dm);CHKERRQ(ierr);

    if (!type) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The particle type must be specified");
    } else if (strcmp(type, PARTICLETRACER) == 0) {
        return ParticleTracerCreate(*particles, flow);
    }
    { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown particle type"); }
}

PETSC_EXTERN PetscErrorCode ParticleSetExactSolution(Particles particles, PetscErrorCode (*exactSolution)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *)) {
    particles->exactSolution = exactSolution;
    return 0;
}

PETSC_EXTERN PetscErrorCode ParticleSetupIntegrator(Particles particles, TS particleTs, TS flowTs) {
    return particles->setupIntegrator(particles, particleTs, flowTs);
}

PETSC_EXTERN PetscErrorCode ParticleDestroy(Particles* particles) {
    PetscErrorCode ierr = (*particles)->destroy(*particles);CHKERRQ(ierr);
    ierr = DMDestroy(&(*particles)->dm);CHKERRQ(ierr);
    free(*particles);
    particles = NULL;
    return 0;
}



//PETSC_EXTERN PetscErrorCode FlowSetupDiscretization(Flow flow) { return flow->setupDiscretization(flow); }
//
//PETSC_EXTERN PetscErrorCode FlowInitialization(Flow flow, DM dm, Vec u) {
//    if (flow->completeFlowInitialization) {
//        return flow->completeFlowInitialization(dm, u);
//    } else {
//        return 0;
//    }
//}
//
//PETSC_EXTERN PetscErrorCode FlowStartProblemSetup(Flow flow) { return flow->startProblemSetup(flow); }
//PETSC_EXTERN PetscErrorCode FlowCompleteProblemSetup(Flow flow, TS ts) { return flow->completeProblemSetup(flow, ts); }
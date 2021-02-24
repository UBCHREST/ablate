#include "particles.h"

PetscErrorCode ParticleCreate(Particles* particles, DM flowDM, ParticleInitializer particleInitializer) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    *particles = malloc(sizeof(struct _Particles));

    // initialize all fields
    (*particles)->flowDM = flowDM;
    (*particles)->dm = NULL;
    (*particles)->parameters = NULL;
    (*particles)->data = NULL;
    (*particles)->exactSolution = NULL;
    (*particles)->particleInitializer = particleInitializer;

    // create and associate the dm
    ierr = DMCreate(PETSC_COMM_WORLD, &(*particles)->dm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)((*particles)->dm), "particles_");CHKERRQ(ierr);
    ierr = DMSetType((*particles)->dm, DMSWARM);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) (*particles)->dm, "Particles");CHKERRQ(ierr);

    // get the dimensions of the flow and set for the particle dm
    PetscInt dim;
    ierr = DMGetDimension(flowDM, &dim);CHKERRQ(ierr);
    ierr = DMSetDimension((*particles)->dm, dim);CHKERRQ(ierr);
    ierr = DMSwarmSetCellDM((*particles)->dm, flowDM);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode ParticleDestroy(Particles* particles) {
    PetscErrorCode ierr = DMDestroy(&(*particles)->dm);CHKERRQ(ierr);
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
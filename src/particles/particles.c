#include "incompressibleFlow.h"
#include "lowMachFlow.h"
#include "particles.h"

PetscErrorCode ParticleCreate(Particles* particles, Flow flow) {
    PetscErrorCode ierr;
    *particles = malloc(sizeof(struct _Particles));

    // create and associate the dm
    ierr = DMCreate(PETSC_COMM_WORLD, &(*particles)->dm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((*particles)->dm, "particles_");CHKERRQ(ierr);

    return IncompressibleFlowCreate(*flow);
}

PETSC_EXTERN PetscErrorCode FlowDestroy(Flow* flow) {
    PetscErrorCode ierr = (*flow)->destroy(*flow);CHKERRQ(ierr);
    ierr = PetscBagDestroy(&(*flow)->parameters);CHKERRQ(ierr);
    free(*flow);
    flow = NULL;
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
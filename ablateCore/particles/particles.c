#include "particles.h"

PetscErrorCode ParticleCreate(ParticleData *particles, PetscInt ndims) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    *particles = malloc(sizeof(struct _ParticleData));

    // initialize all fields
    (*particles)->dm = NULL;
    (*particles)->parameters = NULL;
    (*particles)->data = NULL;
    (*particles)->exactSolution = NULL;
    (*particles)->timeInitial = 0.0;
    (*particles)->timeFinal = 0.0;

    // create and associate the dm
    ierr = DMCreate(PETSC_COMM_WORLD, &(*particles)->dm);CHKERRQ(ierr);
    ierr = DMSetType((*particles)->dm, DMSWARM);CHKERRQ(ierr);
    ierr = DMSetDimension((*particles)->dm, ndims);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ParticleInitializeFlow(ParticleData particles, FlowData flowData) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // get the dimensions of the flow and make sure it is the same as particles
    ierr = DMSwarmSetCellDM(particles->dm, flowData->dm);CHKERRQ(ierr);

    // Store the values in the particles from the ts and flow
    particles->flowFinal = flowData->flowField;
    ierr = VecDuplicate(particles->flowFinal, &(particles->flowInitial));CHKERRQ(ierr);
    ierr = VecCopy(flowData->flowField, particles->flowInitial);CHKERRQ(ierr);

    // Find the velocity field
    PetscBool found;
    ierr = PetscEListFind(flowData->numberFlowFields,flowData->flowFieldNames, "velocity",&(particles->flowVelocityFieldIndex),&found);CHKERRQ(ierr);
    if(!found){
        // get the particle data comm
        MPI_Comm comm;
        ierr = PetscObjectGetComm((PetscObject) particles->dm, &comm);CHKERRQ(ierr);
        SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE,"unable to find velocity in flowData");
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ParticleSetExactSolutionFlow(ParticleData particles, PetscErrorCode (*exactSolution)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *),
                                            void *exactSolutionContext) {
    PetscFunctionBeginUser;
    particles->exactSolution = exactSolution;
    particles->exactSolutionContext = exactSolutionContext;
    PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode ParticleDestroy(ParticleData *particles) {
    PetscErrorCode ierr = DMDestroy(&(*particles)->dm);CHKERRQ(ierr);
    free(*particles);
    particles = NULL;
    return 0;
}
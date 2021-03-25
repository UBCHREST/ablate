#include "particles.h"
#include <petscviewerhdf5.h>

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
    (*particles)->dmChanged = PETSC_FALSE;

    // create and associate the dm
    ierr = DMCreate(PETSC_COMM_WORLD, &(*particles)->dm);CHKERRQ(ierr);
    ierr = DMSetType((*particles)->dm, DMSWARM);CHKERRQ(ierr);
    ierr = DMSetDimension((*particles)->dm, ndims);CHKERRQ(ierr);

    // setup the basic field names
    (*particles)->numberFields = 2;
    ierr = PetscMalloc1((*particles)->numberFields, &((*particles)->fieldDescriptors));CHKERRQ(ierr);

    // set the basic names
    PetscStrallocpy(DMSwarmPICField_coor, (char **)&(*particles)->fieldDescriptors[0].fieldName);
    (*particles)->fieldDescriptors[0].type = PETSC_DOUBLE;
    (*particles)->fieldDescriptors[0].components = ndims;
    PetscStrallocpy(DMSwarmField_pid, (char **)&(*particles)->fieldDescriptors[1].fieldName);
    (*particles)->fieldDescriptors[1].type = PETSC_INT64;
    (*particles)->fieldDescriptors[1].components = 1;
    PetscFunctionReturn(0);
}

PetscErrorCode ParticleInitializeFlow(ParticleData particles, FlowData flowData) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // before setting up the flow finalize the fields
    ierr = DMSwarmFinalizeFieldRegister(particles->dm);CHKERRQ(ierr);

    // get the dimensions of the flow and make sure it is the same as particles
    ierr = DMSwarmSetCellDM(particles->dm, flowData->dm);CHKERRQ(ierr);

    // Store the values in the particles from the ts and flow
    particles->flowFinal = flowData->flowField;
    ierr = VecDuplicate(particles->flowFinal, &(particles->flowInitial));CHKERRQ(ierr);
    ierr = VecCopy(flowData->flowField, particles->flowInitial);CHKERRQ(ierr);

    // Find the velocity field
    PetscBool found;
    for (PetscInt f =0; f < flowData->numberFlowFields; f++){
        ierr = PetscStrcmp("velocity",flowData->flowFieldDescriptors[f].fieldName, &found);CHKERRQ(ierr);
        if (found){
            particles->flowVelocityFieldIndex = f;
            break;
        }
    }
    if (!found){
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

PETSC_EXTERN PetscErrorCode ParticleRegisterPetscDatatypeField(ParticleData particles, const char *fieldname, PetscInt blocksize, PetscDataType type) {
    PetscFunctionBeginUser;
    // add the value to the field
    PetscErrorCode ierr = DMSwarmRegisterPetscDatatypeField(particles->dm, fieldname, blocksize, type);CHKERRQ(ierr);

    // store the field
    particles->numberFields++;
    if (particles->fieldDescriptors == NULL){
        ierr = PetscMalloc1(particles->numberFields, &(particles->fieldDescriptors));CHKERRQ(ierr);
    }else{
        ierr = PetscRealloc(sizeof(ParticleFieldDescriptor)*particles->numberFields,&(particles->fieldDescriptors));CHKERRQ(ierr);
    }
    particles->fieldDescriptors[particles->numberFields-1].type = type;
    particles->fieldDescriptors[particles->numberFields-1].components = blocksize;
    PetscStrallocpy(fieldname, (char **)&(particles->fieldDescriptors[particles->numberFields-1].fieldName));
    PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode ParticleDestroy(ParticleData *particles) {
    // remove each allocated string and
    for (PetscInt i =0; i < (*particles)->numberFields; i++){
        PetscFree((*particles)->fieldDescriptors[i].fieldName);
    }
    PetscFree((*particles)->fieldDescriptors);

    PetscErrorCode ierr = DMDestroy(&(*particles)->dm);CHKERRQ(ierr);
    free(*particles);
    particles = NULL;
    return 0;
}

static PetscErrorCode DMSequenceViewTimeHDF5(DM dm, PetscViewer viewer)
{
    Vec            stamp;
    PetscMPIInt    rank;
    PetscErrorCode ierr;

    PetscFunctionBegin;

    // get the seqnum and value from the dm
    PetscInt seqnum;
    PetscReal value;
    ierr =  DMGetOutputSequenceNumber(dm, &seqnum, &value);CHKERRMPI(ierr);

    if (seqnum < 0){
        PetscFunctionReturn(0);
    }
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRMPI(ierr);
    ierr = VecCreateMPI(PetscObjectComm((PetscObject) viewer), rank ? 0 : 1, 1, &stamp);CHKERRQ(ierr);
    ierr = VecSetBlockSize(stamp, 1);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) stamp, "time");CHKERRQ(ierr);
    if (!rank) {
        ierr = VecSetValue(stamp, 0, value, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(stamp);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(stamp);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PushGroup(viewer, "/");CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(viewer, seqnum);CHKERRQ(ierr);
    ierr = VecView(stamp, viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&stamp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ParticleView(ParticleData particleData, PetscViewer viewer) {
    PetscFunctionBegin;
    Vec            particleVector;
    PetscErrorCode ierr;

    for (PetscInt f =0; f < particleData->numberFields; f++){
        if (particleData->fieldDescriptors[f].type == PETSC_DOUBLE) {
            ierr = DMSwarmCreateGlobalVectorFromField(particleData->dm, particleData->fieldDescriptors[f].fieldName, &particleVector);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject)particleVector, particleData->fieldDescriptors[f].fieldName);CHKERRQ(ierr);
            ierr = VecView(particleVector, viewer);CHKERRQ(ierr);
            ierr = DMSwarmDestroyGlobalVectorFromField(particleData->dm, particleData->fieldDescriptors[f].fieldName, &particleVector);CHKERRQ(ierr);
        }
    }

    // if this is an hdf5Viewer
    PetscBool ishdf5;
    ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
    if (ishdf5){
        DMSequenceViewTimeHDF5(particleData->dm, viewer);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ParticleViewFromOptions(ParticleData particleData,PetscObject obj, char *title) {
    PetscFunctionBegin;
    Vec            particleVector;
    PetscErrorCode ierr;

    for (PetscInt f =0; f < particleData->numberFields; f++){
        if (particleData->fieldDescriptors[f].type == PETSC_DOUBLE) {
            ierr = DMSwarmCreateGlobalVectorFromField(particleData->dm, particleData->fieldDescriptors[f].fieldName, &particleVector);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject)particleVector, particleData->fieldDescriptors[f].fieldName);CHKERRQ(ierr);
            ierr = VecViewFromOptions(particleVector, obj, title);CHKERRQ(ierr);
            ierr = DMSwarmDestroyGlobalVectorFromField(particleData->dm, particleData->fieldDescriptors[f].fieldName, &particleVector);CHKERRQ(ierr);
        }
    }

    PetscFunctionReturn(0);
}
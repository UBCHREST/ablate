#include "flow.h"

PetscErrorCode FlowCreate(FlowData* flow) {
    PetscFunctionBeginUser;
    *flow = malloc(sizeof(struct _FlowData));

    // initialize all fields
    (*flow)->dm = NULL;
    (*flow)->data = NULL;
    (*flow)->flowField = NULL;

    // setup the basic field names
    (*flow)->numberFlowFields = 0;
    PetscErrorCode ierr = PetscMalloc1((*flow)->numberFlowFields, &((*flow)->fieldDescriptors));CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterFields(FlowData flow, const char fieldName[],const char fieldPrefix[], PetscInt components) {
    PetscFunctionBeginUser;
    // store the field
    flow->numberFlowFields++;
    PetscErrorCode ierr = PetscRealloc(sizeof(FlowFieldDescriptor)*flow->numberFlowFields,&(flow->fieldDescriptors));CHKERRQ(ierr);
    flow->fieldDescriptors[flow->numberFlowFields-1].components = components;
    PetscStrallocpy(fieldName, (char **)&(flow->fieldDescriptors[flow->numberFlowFields-1].fieldName));
    PetscStrallocpy(fieldPrefix, (char **)&(flow->fieldDescriptors[flow->numberFlowFields-1].fieldPrefix));

    // get the dm prefix to help name the fe objects
    const char *dmPrefix;
    ierr = DMGetOptionsPrefix(flow->dm, &dmPrefix);CHKERRQ(ierr);
    char combinedFieldPrefix[128] = "";

    /* Create finite element */
    ierr = PetscStrlcat(combinedFieldPrefix, dmPrefix, 128);CHKERRQ(ierr);
    ierr = PetscStrlcat(combinedFieldPrefix, fieldPrefix, 128);CHKERRQ(ierr);

    // determine if it a simplex element and the number of dimensions
    DMPolytopeType ct;
    PetscInt cStart;
    ierr = DMPlexGetHeightStratum(flow->dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(flow->dm, cStart, &ct);CHKERRQ(ierr);
    PetscBool simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;

    // Determine the number of dimensions
    PetscInt dim;
    ierr = DMGetDimension(flow->dm, &dim);CHKERRQ(ierr);

    // create a petsc fe
    MPI_Comm comm;
    PetscFE petscFE;
    ierr = PetscObjectGetComm((PetscObject)flow->dm, &comm);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(comm, dim, components, simplex, combinedFieldPrefix, PETSC_DEFAULT, &petscFE);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)petscFE, fieldName);CHKERRQ(ierr);

    //If this is not the first field, copy the quadrature locations
    if(flow->numberFlowFields > 1){
        PetscFE referencePetscFE;
        ierr = DMGetField(flow->dm, 0, NULL, (PetscObject*)&referencePetscFE);CHKERRQ(ierr);
        ierr = PetscFECopyQuadrature(referencePetscFE, petscFE);CHKERRQ(ierr);
    }

    // Store the field and destroy copy
    ierr = DMSetField(flow->dm, flow->numberFlowFields-1, NULL, (PetscObject)petscFE);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&petscFE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode FlowDestroy(FlowData* flow) {
    PetscFunctionBeginUser;
    // remove each allocated string and flow field
    for (PetscInt i =0; i < (*flow)->numberFlowFields; i++){
        PetscFree((*flow)->fieldDescriptors[i].fieldName);
        PetscFree((*flow)->fieldDescriptors[i].fieldPrefix);
    }
    PetscFree((*flow)->fieldDescriptors);

    if ((*flow)->flowField){
        PetscErrorCode ierr = VecDestroy(&((*flow)->flowField));CHKERRQ(ierr);
    }
    free(*flow);
    flow = NULL;
    PetscFunctionReturn(0);
}
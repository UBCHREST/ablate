#include "flow.h"

PetscErrorCode FlowCreate(FlowData* flow) {
    PetscFunctionBeginUser;
    *flow = malloc(sizeof(struct _FlowData));

    // initialize all fields
    (*flow)->dm = NULL;
    (*flow)->auxDm = NULL;
    (*flow)->data = NULL;
    (*flow)->flowField = NULL;
    (*flow)->auxField = NULL;

    // setup the basic field names
    (*flow)->numberFlowFields = 0;
    PetscErrorCode ierr = PetscMalloc1((*flow)->numberFlowFields, &((*flow)->flowFieldDescriptors));CHKERRQ(ierr);
    (*flow)->numberAuxFields = 0;
    ierr = PetscMalloc1((*flow)->numberAuxFields, &((*flow)->auxFieldDescriptors));CHKERRQ(ierr);

    // setup empty update fields
    (*flow)->numberPreStepFunctions = 0;
    ierr = PetscMalloc1((*flow)->numberPreStepFunctions, &((*flow)->preStepFunctions));CHKERRQ(ierr);
    (*flow)->numberPostStepFunctions = 0;
    ierr = PetscMalloc1((*flow)->numberPostStepFunctions, &((*flow)->postStepFunctions));CHKERRQ(ierr);


    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterField(FlowData flow, const char *fieldName, const char *fieldPrefix, PetscInt components) {
    PetscFunctionBeginUser;
    // store the field
    flow->numberFlowFields++;
    PetscErrorCode ierr = PetscRealloc(sizeof(FlowFieldDescriptor)*flow->numberFlowFields,&(flow->flowFieldDescriptors));CHKERRQ(ierr);
    flow->flowFieldDescriptors[flow->numberFlowFields-1].components = components;
    PetscStrallocpy(fieldName, (char **)&(flow->flowFieldDescriptors[flow->numberFlowFields-1].fieldName));
    PetscStrallocpy(fieldPrefix, (char **)&(flow->flowFieldDescriptors[flow->numberFlowFields-1].fieldPrefix));

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

PetscErrorCode FlowRegisterAuxField(FlowData flow, const char *fieldName, const char *fieldPrefix, PetscInt components) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // check to see if need to create an aux dm
    if(flow->auxDm == NULL){
        /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
        DM coordDM;
        ierr = DMGetCoordinateDM(flow->dm, &coordDM);CHKERRQ(ierr);
        ierr = DMClone(flow->dm, &(flow->auxDm));CHKERRQ(ierr);

        // this is a hard coded "dmAux" that petsc looks for
        ierr = PetscObjectCompose((PetscObject) flow->dm, "dmAux", (PetscObject) flow->auxField);CHKERRQ(ierr);
        ierr = DMSetCoordinateDM(flow->auxDm, coordDM);CHKERRQ(ierr);
    }

    // store the field
    flow->numberAuxFields++;
    ierr = PetscRealloc(sizeof(FlowFieldDescriptor)*flow->numberAuxFields,&(flow->auxFieldDescriptors));CHKERRQ(ierr);
    flow->auxFieldDescriptors[flow->numberAuxFields-1].components = components;
    PetscStrallocpy(fieldName, (char **)&(flow->auxFieldDescriptors[flow->numberAuxFields-1].fieldName));
    PetscStrallocpy(fieldPrefix, (char **)&(flow->auxFieldDescriptors[flow->numberAuxFields-1].fieldPrefix));

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
    ierr = DMSetField(flow->auxDm, flow->numberAuxFields-1, NULL, (PetscObject)petscFE);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&petscFE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FlowFinalizeRegisterFields(FlowData flow){
    PetscFunctionBeginUser;
    // Create the discrete systems for the DM based upon the fields added to the DM
    PetscErrorCode ierr = DMCreateDS(flow->dm);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode FlowTSPreStepFunction(TS ts){
    PetscFunctionBeginUser;
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    FlowData flowData;
    ierr = DMGetApplicationContext(dm, &flowData);CHKERRQ(ierr);

    for(PetscInt i =0; i < flowData->numberPreStepFunctions; i++){
        ierr = flowData->preStepFunctions[i].updateFunction(ts, flowData->preStepFunctions[i].context);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

static PetscErrorCode FlowTSPostStepFunction(TS ts){
    PetscFunctionBeginUser;
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    FlowData flowData;
    ierr = DMGetApplicationContext(dm, &flowData);CHKERRQ(ierr);

    for(PetscInt i =0; i < flowData->numberPostStepFunctions; i++){
        ierr = flowData->postStepFunctions[i].updateFunction(ts, flowData->postStepFunctions[i].context);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode FlowCompleteProblemSetup(FlowData flowData, TS ts){
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBeginUser;
    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm, &(flowData->flowField));CHKERRQ(ierr);

    if(flowData->auxDm){
        ierr = DMCreateDS(flowData->auxDm);CHKERRQ(ierr);
        ierr = DMCreateLocalVector(flowData->auxDm, &(flowData->auxField));CHKERRQ(ierr);

        // attach this field as aux vector to the dm
        ierr = PetscObjectCompose((PetscObject) flowData->dm, "A", (PetscObject) flowData->auxField);CHKERRQ(ierr);
    }

    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL);CHKERRQ(ierr);
    ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL);CHKERRQ(ierr);

    ierr = TSSetPreStep(ts, FlowTSPreStepFunction);CHKERRQ(ierr);
    ierr = TSSetPostStep(ts, FlowTSPostStepFunction);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterPreStep(FlowData flowData, PetscErrorCode (*updateFunction)(TS ts, void* context), void* context) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    flowData->numberPreStepFunctions++;
    ierr = PetscRealloc(sizeof(FlowUpdateFunction)*flowData->numberPreStepFunctions,&(flowData->preStepFunctions));CHKERRQ(ierr);
    flowData->preStepFunctions[flowData->numberPreStepFunctions-1].updateFunction = updateFunction;
    flowData->preStepFunctions[flowData->numberPreStepFunctions-1].context = context;
    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterPostStep(FlowData flowData, PetscErrorCode (*updateFunction)(TS ts, void* context), void* context) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    flowData->numberPostStepFunctions++;
    ierr = PetscRealloc(sizeof(FlowUpdateFunction)*flowData->numberPostStepFunctions,&(flowData->postStepFunctions));CHKERRQ(ierr);
    flowData->postStepFunctions[flowData->numberPostStepFunctions-1].updateFunction = updateFunction;
    flowData->postStepFunctions[flowData->numberPostStepFunctions-1].context = context;
    PetscFunctionReturn(0);
}

PetscErrorCode FlowDestroy(FlowData* flow) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    // remove each allocated string and flow field
    for (PetscInt i =0; i < (*flow)->numberFlowFields; i++){
        PetscFree((*flow)->flowFieldDescriptors[i].fieldName);
        PetscFree((*flow)->flowFieldDescriptors[i].fieldPrefix);
    }
    PetscFree((*flow)->flowFieldDescriptors);

    if ((*flow)->auxField){
        ierr = VecDestroy(&((*flow)->auxField));CHKERRQ(ierr);
    }

    if ((*flow)->auxDm){
        ierr = DMDestroy(&((*flow)->auxDm));CHKERRQ(ierr);
    }

    if ((*flow)->flowField){
        ierr = VecDestroy(&((*flow)->flowField));CHKERRQ(ierr);
    }

    PetscFree((*flow)->preStepFunctions);
    PetscFree((*flow)->postStepFunctions);

    free(*flow);
    flow = NULL;
    PetscFunctionReturn(0);
}
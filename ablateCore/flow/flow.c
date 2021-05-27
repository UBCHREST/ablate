#include "flow.h"

static PetscBool flowInitialized = PETSC_FALSE;
static PetscFunctionList flowSetupFunctionList = NULL;

PetscErrorCode FlowCreate(FlowData* flow) {
    PetscFunctionBeginUser;
    *flow = malloc(sizeof(struct _FlowData));

    // initialize all fields
    (*flow)->type = NULL;
    (*flow)->dm = NULL;
    (*flow)->auxDm = NULL;
    (*flow)->data = NULL;
    (*flow)->flowField = NULL;
    (*flow)->auxField = NULL;
    (*flow)->options = NULL;

    // set default methods to null
    (*flow)->flowSetupDiscretization = NULL;
    (*flow)->flowStartProblemSetup = NULL;
    (*flow)->flowCompleteProblemSetup = NULL;
    (*flow)->flowCompleteFlowInitialization = NULL;
    (*flow)->flowDestroy = NULL;

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

PetscErrorCode FlowSetFromOptions_LowMachFlow(FlowData);
PetscErrorCode FlowSetFromOptions_IncompressibleFlow(FlowData);

static PetscErrorCode FlowInitialize(){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if (!flowInitialized){
        flowInitialized = PETSC_TRUE;
        ierr = FlowRegister("lowMach", FlowSetFromOptions_LowMachFlow);CHKERRQ(ierr);
        ierr = FlowRegister("incompressible", FlowSetFromOptions_IncompressibleFlow);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegister(const char* name,const FlowSetupFunction function){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if (!flowInitialized) {
        ierr = FlowInitialize();CHKERRQ(ierr);
    }
    ierr = PetscFunctionListAdd(&flowSetupFunctionList,name,function);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FlowSetType(FlowData flow, const char* type){
    PetscFunctionBeginUser;
    PetscErrorCode ierr = PetscStrallocpy(type, &flow->type);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FlowSetOptions(FlowData flow, PetscOptions options){
    PetscFunctionBeginUser;
    flow->options = options;
    PetscFunctionReturn(0);
}

PetscErrorCode FlowSetFromOptions(FlowData flow){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ierr = FlowInitialize();CHKERRQ(ierr);

    // get the
    const char ** typeList;
    PetscInt numberTypes;
    ierr = PetscFunctionListGet(flowSetupFunctionList,&typeList, &numberTypes);CHKERRQ(ierr);

    // get the value from the options
    PetscInt value;
    PetscBool set;
    ierr = PetscOptionsGetEList(flow->options, NULL, "-flowType", typeList, numberTypes,&value,&set);CHKERRQ(ierr);
    if (!set && !flow->type){
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_TYPENOTSET, "The Fow type must be set with EOSSetType() or -eosType");
    }
    if (set){
        ierr =  PetscStrallocpy(typeList[value], (char**) &flow->type);CHKERRQ(ierr);
    }

    // Get the create function from the function list
    FlowSetupFunction setupFunction;
    ierr = PetscFunctionListFind(flowSetupFunctionList,flow->type, &setupFunction);CHKERRQ(ierr);

    // call and setup the eos
    ierr = setupFunction(flow);CHKERRQ(ierr);

    PetscFree(typeList);
    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterField(FlowData flow, const char* fieldName, const char* fieldPrefix, PetscInt components, enum FieldType fieldType) {
    PetscFunctionBeginUser;
    // store the field
    flow->numberFlowFields++;
    PetscErrorCode ierr;
    if (flow->flowFieldDescriptors == NULL){
        ierr = PetscMalloc1(flow->numberFlowFields, &(flow->flowFieldDescriptors));CHKERRQ(ierr);
    }else{
        ierr = PetscRealloc(sizeof(FlowFieldDescriptor)*flow->numberFlowFields,&(flow->flowFieldDescriptors));CHKERRQ(ierr);
    }
    flow->flowFieldDescriptors[flow->numberFlowFields-1].components = components;
    flow->flowFieldDescriptors[flow->numberFlowFields-1].fieldType = fieldType;
    PetscStrallocpy(fieldName, (char **)&(flow->flowFieldDescriptors[flow->numberFlowFields-1].fieldName));
    PetscStrallocpy(fieldPrefix, (char **)&(flow->flowFieldDescriptors[flow->numberFlowFields-1].fieldPrefix));

    // get the dm prefix to help name the fe objects
    const char *dmPrefix;
    ierr = DMGetOptionsPrefix(flow->dm, &dmPrefix);CHKERRQ(ierr);
    char combinedFieldPrefix[128] = "";

    // extract the object comm
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)flow->dm, &comm);CHKERRQ(ierr);

    /* Create finite element */
    ierr = PetscStrlcat(combinedFieldPrefix, dmPrefix, 128);CHKERRQ(ierr);
    ierr = PetscStrlcat(combinedFieldPrefix, fieldPrefix, 128);CHKERRQ(ierr);

    // Determine the number of dimensions
    PetscInt dim;
    ierr = DMGetDimension(flow->dm, &dim);CHKERRQ(ierr);

    switch (fieldType) {
        case FE:{
            // determine if it a simplex element and the number of dimensions
            DMPolytopeType ct;
            PetscInt cStart;
            ierr = DMPlexGetHeightStratum(flow->dm, 0, &cStart, NULL);CHKERRQ(ierr);
            ierr = DMPlexGetCellType(flow->dm, cStart, &ct);CHKERRQ(ierr);
            PetscInt simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
            PetscInt simplexGlobal;

            // Assume true if any rank says true
            ierr = MPI_Allreduce(&simplex, &simplexGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);

            // create a petsc fe
            PetscFE petscFE;
            ierr = PetscFECreateDefault(comm, dim, components, simplexGlobal? PETSC_TRUE : PETSC_FALSE, combinedFieldPrefix, PETSC_DEFAULT, &petscFE);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject)petscFE, fieldName);CHKERRQ(ierr);

            //If this is not the first field, copy the quadrature locations
            if (flow->numberFlowFields > 1){
                PetscFE referencePetscFE;
                ierr = DMGetField(flow->dm, 0, NULL, (PetscObject*)&referencePetscFE);CHKERRQ(ierr);
                ierr = PetscFECopyQuadrature(referencePetscFE, petscFE);CHKERRQ(ierr);
            }

            // Store the field and destroy copy
            ierr = DMSetField(flow->dm, flow->numberFlowFields-1, NULL, (PetscObject)petscFE);CHKERRQ(ierr);
            ierr = PetscFEDestroy(&petscFE);CHKERRQ(ierr);
        }break;
        case FV:{
            PetscFV           fvm;
            ierr = PetscFVCreate(PETSC_COMM_WORLD, &fvm);CHKERRQ(ierr);
            ierr = PetscObjectSetOptionsPrefix((PetscObject) fvm, combinedFieldPrefix);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) fvm, fieldName);CHKERRQ(ierr);

            ierr = PetscFVSetFromOptions(fvm);CHKERRQ(ierr);
            ierr = PetscFVSetNumComponents(fvm, components);CHKERRQ(ierr);
            ierr = PetscFVSetSpatialDimension(fvm, dim);CHKERRQ(ierr);

            ierr = DMSetField(flow->dm, flow->numberFlowFields-1, NULL, (PetscObject)fvm);CHKERRQ(ierr);
            ierr = PetscFVDestroy(&fvm);CHKERRQ(ierr);
        }break;
        default:{
            SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Unknown field type for flow");
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterAuxField(FlowData flow, const char *fieldName, const char *fieldPrefix, PetscInt components, enum FieldType fieldType) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // check to see if need to create an aux dm
    if (flow->auxDm == NULL){
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
    if (flow->auxFieldDescriptors == NULL){
        ierr = PetscMalloc1(flow->numberAuxFields, &(flow->auxFieldDescriptors));CHKERRQ(ierr);
    }else{
        ierr = PetscRealloc(sizeof(FlowFieldDescriptor)*flow->numberAuxFields,&(flow->auxFieldDescriptors));CHKERRQ(ierr);
    }
    flow->auxFieldDescriptors[flow->numberAuxFields-1].components = components;
    flow->auxFieldDescriptors[flow->numberAuxFields-1].fieldType = fieldType;
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
    PetscInt simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct) + 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt simplexGlobal;

    // extract the object comm
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)flow->dm, &comm);CHKERRQ(ierr);

    // Assume true if any rank says true
    ierr = MPI_Allreduce(&simplex, &simplexGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);

    // Determine the number of dimensions
    PetscInt dim;
    ierr = DMGetDimension(flow->dm, &dim);CHKERRQ(ierr);

    switch (fieldType) {
        case FE: {
            // create a petsc fe
            PetscFE petscFE;
            ierr = PetscFECreateDefault(comm, dim, components, simplexGlobal? PETSC_TRUE : PETSC_FALSE, combinedFieldPrefix, PETSC_DEFAULT, &petscFE);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject)petscFE, fieldName);CHKERRQ(ierr);

            //If this is not the first field, copy the quadrature locations
            if (flow->numberFlowFields > 1){
                PetscFE referencePetscFE;
                ierr = DMGetField(flow->dm, 0, NULL, (PetscObject*)&referencePetscFE);CHKERRQ(ierr);
                ierr = PetscFECopyQuadrature(referencePetscFE, petscFE);CHKERRQ(ierr);
            }

            // Store the field and destroy copy
            ierr = DMSetField(flow->auxDm, flow->numberAuxFields-1, NULL, (PetscObject)petscFE);CHKERRQ(ierr);
            ierr = PetscFEDestroy(&petscFE);CHKERRQ(ierr);
        }break;
        case FV:{
            PetscFV           fvm;
            ierr = PetscFVCreate(PETSC_COMM_WORLD, &fvm);CHKERRQ(ierr);
            ierr = PetscObjectSetOptionsPrefix((PetscObject) fvm, combinedFieldPrefix);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) fvm, fieldName);CHKERRQ(ierr);

            ierr = PetscFVSetFromOptions(fvm);CHKERRQ(ierr);
            ierr = PetscFVSetNumComponents(fvm, components);CHKERRQ(ierr);
            ierr = PetscFVSetSpatialDimension(fvm, dim);CHKERRQ(ierr);

            ierr = DMSetField(flow->auxDm, flow->numberAuxFields-1, NULL, (PetscObject)fvm);CHKERRQ(ierr);
            ierr = PetscFVDestroy(&fvm);CHKERRQ(ierr);
        }break;
            default:{
            SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Unknown field type for flow");
        }
    }

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

    for (PetscInt i =0; i < flowData->numberPreStepFunctions; i++){
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

    for (PetscInt i =0; i < flowData->numberPostStepFunctions; i++){
        ierr = flowData->postStepFunctions[i].updateFunction(ts, flowData->postStepFunctions[i].context);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode FlowSetupDiscretization(FlowData flowData, DM* dm){
    PetscFunctionBeginUser;
    if(flowData->flowSetupDiscretization){
        PetscErrorCode ierr = flowData->flowSetupDiscretization(flowData, dm);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode FlowStartProblemSetup(FlowData flowData){
    PetscFunctionBeginUser;
    if(flowData->flowStartProblemSetup){
        PetscErrorCode ierr = flowData->flowStartProblemSetup(flowData);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode FlowCompleteProblemSetup(FlowData flowData, TS ts){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Call the class specific implementation
    if(flowData->flowCompleteProblemSetup){
        PetscErrorCode ierr = flowData->flowCompleteProblemSetup(flowData, ts);CHKERRQ(ierr);
    }

    // Setup the solve with the ts
    DM dm = flowData->dm;
    ierr = TSSetDM(ts, flowData->dm);CHKERRQ(ierr);

    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm, &(flowData->flowField));CHKERRQ(ierr);

    if (flowData->auxDm){
        ierr = DMCreateDS(flowData->auxDm);CHKERRQ(ierr);
        ierr = DMCreateLocalVector(flowData->auxDm, &(flowData->auxField));CHKERRQ(ierr);

        // attach this field as aux vector to the dm
        ierr = PetscObjectCompose((PetscObject) flowData->dm, "A", (PetscObject) flowData->auxField);CHKERRQ(ierr);

        ierr = PetscObjectSetName((PetscObject)flowData->auxField, "auxField");CHKERRQ(ierr);
    }

    // Check if any of the fields are fe
    PetscBool isFE = PETSC_FALSE;
    PetscBool isFV = PETSC_FALSE;
    for (PetscInt f =0; f < flowData->numberFlowFields; f++){
        switch(flowData->flowFieldDescriptors[f].fieldType){
            case(FE):
                isFE = PETSC_TRUE;
                break;
            case(FV):
                isFV = PETSC_TRUE;
                break;
            default:
                SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG,"Unknown field type for flow");
        }
    }

    if (isFE) {
        ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, NULL);CHKERRQ(ierr);
        ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, NULL);CHKERRQ(ierr);
        ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, NULL);CHKERRQ(ierr);
    }
    if (isFV){
        ierr = DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, flowData);CHKERRQ(ierr);
    }
    ierr = TSSetPreStep(ts, FlowTSPreStepFunction);CHKERRQ(ierr);
    ierr = TSSetPostStep(ts, FlowTSPostStepFunction);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode FlowCompleteFlowInitialization(FlowData flowData, DM dm, Vec u){
    PetscFunctionBeginUser;
    if(flowData->flowCompleteFlowInitialization){
        PetscErrorCode ierr = flowData->flowCompleteFlowInitialization(dm, u);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterPreStep(FlowData flowData, PetscErrorCode (*updateFunction)(TS ts, void* context), void* context) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    flowData->numberPreStepFunctions++;
    if (flowData->preStepFunctions == NULL){
        ierr = PetscMalloc1(flowData->numberPreStepFunctions, &(flowData->preStepFunctions));CHKERRQ(ierr);
    }else{
        ierr = PetscRealloc(sizeof(FlowUpdateFunction)*flowData->numberPreStepFunctions,&(flowData->preStepFunctions));CHKERRQ(ierr);
    }
    flowData->preStepFunctions[flowData->numberPreStepFunctions-1].updateFunction = updateFunction;
    flowData->preStepFunctions[flowData->numberPreStepFunctions-1].context = context;
    PetscFunctionReturn(0);
}

PetscErrorCode FlowRegisterPostStep(FlowData flowData, PetscErrorCode (*updateFunction)(TS ts, void* context), void* context) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    flowData->numberPostStepFunctions++;
    if (flowData->postStepFunctions == NULL){
        ierr = PetscMalloc1(flowData->numberPostStepFunctions, &(flowData->postStepFunctions));CHKERRQ(ierr);
    }else{
        ierr = PetscRealloc(sizeof(FlowUpdateFunction)*flowData->numberPostStepFunctions,&(flowData->postStepFunctions));CHKERRQ(ierr);
    }
    flowData->postStepFunctions[flowData->numberPostStepFunctions-1].updateFunction = updateFunction;
    flowData->postStepFunctions[flowData->numberPostStepFunctions-1].context = context;
    PetscFunctionReturn(0);
}


PetscErrorCode FlowView(FlowData flowData,PetscViewer viewer) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    // Always save the main flowField
    ierr = VecView(flowData->flowField, viewer);CHKERRQ(ierr);

    //If there is aux data output
    if (flowData->auxField) {
        // copy over the sequence data from the main dm
        PetscReal dmTime;
        PetscInt dmSequence;
        ierr = DMGetOutputSequenceNumber(flowData->dm, &dmSequence, &dmTime);CHKERRQ(ierr);
        ierr = DMSetOutputSequenceNumber(flowData->auxDm, dmSequence, dmTime);CHKERRQ(ierr);

        Vec auxGlobalField;
        ierr = DMGetGlobalVector(flowData->auxDm, &auxGlobalField);CHKERRQ(ierr);

        // copy over the name of the auxFieldVector
        const char *name;
        ierr = PetscObjectGetName((PetscObject)flowData->auxField, &name);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)auxGlobalField, name);CHKERRQ(ierr);

        ierr = DMLocalToGlobal(flowData->auxDm, flowData->auxField, INSERT_VALUES, auxGlobalField);CHKERRQ(ierr);
        ierr = VecView(auxGlobalField, viewer);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(flowData->auxDm, &auxGlobalField);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode FlowViewFromOptions(FlowData flowData, char *optionName) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    PetscViewer viewer;
    PetscBool         viewerCreated;

    // generate a petsc viewer from the options provided
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)flowData->dm),NULL,NULL,optionName,&viewer, NULL,&viewerCreated);CHKERRQ(ierr);
    if (viewerCreated){
        ierr = FlowView(flowData, viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode FlowDestroy(FlowData* flow) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if((*flow)->flowDestroy){
        ierr = (*flow)->flowDestroy(*flow);CHKERRQ(ierr);
    }

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

    if ((*flow)->flowField){
        PetscFree((*flow)->data);
    }

    free(*flow);
    flow = NULL;
    PetscFunctionReturn(0);
}


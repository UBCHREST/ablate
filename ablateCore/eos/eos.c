#include "eos.h"
#include "perfectGas.h"

static PetscBool eosInitialized = PETSC_FALSE;
static PetscFunctionList eosSetupFunctionList = NULL;

static PetscErrorCode EOSInitialize(){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if (!eosInitialized){
        eosInitialized = PETSC_TRUE;
        ierr = EOSRegister("perfectGas", EOSSetFromOptions_PerfectGas);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode EOSRegister(const char* name,const EOSSetupFunction function){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if (!eosInitialized) {
        ierr = EOSInitialize();CHKERRQ(ierr);
    }
    ierr = PetscFunctionListAdd(&eosSetupFunctionList,name,function);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode EOSCreate(EOSData* eos) {
    PetscFunctionBeginUser;
    *eos = malloc(sizeof(struct _EOSData));

    // initialize all fields
    (*eos)->options = NULL;
    (*eos)->data = NULL;
    (*eos)->type = NULL;

    // set methods to null
    (*eos)->eosView = NULL;
    (*eos)->eosDestroy = NULL;
    (*eos)->eosDecodeState = NULL;

    PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode EOSSetType(EOSData eos, const char* type) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr = PetscStrallocpy(type, &eos->type);CHKERRQ(ierr);
    return 0;
}

PETSC_EXTERN PetscErrorCode EOSSetOptions(EOSData eos, PetscOptions options){
    PetscFunctionBeginUser;
    eos->options = options;
    PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode EOSSetFromOptions(EOSData eos){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    ierr = EOSInitialize();CHKERRQ(ierr);

    // get the
    const char ** typeList;
    PetscInt numberTypes;
    ierr = PetscFunctionListGet(eosSetupFunctionList,&typeList, &numberTypes);CHKERRQ(ierr);

    // get the value from the options
    PetscInt value;
    PetscBool set;
    ierr = PetscOptionsGetEList(eos->options, NULL, "-eosType", typeList, numberTypes,&value,&set);CHKERRQ(ierr);
    if (!set && !eos->type){
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_TYPENOTSET, "The EOS type must be set with EOSSetType() or -eosType");
    }
    if (set){
        ierr =  PetscStrallocpy(typeList[value], (char**) &eos->type);CHKERRQ(ierr);
    }

    // Get the create function from the function list
    EOSSetupFunction setupFunction;
    ierr = PetscFunctionListFind(eosSetupFunctionList,eos->type, &setupFunction);CHKERRQ(ierr);

    // call and setup the eos
    ierr = setupFunction(eos);CHKERRQ(ierr);

    PetscFree(typeList);
    PetscFunctionReturn(0);
}

PetscErrorCode EOSView(EOSData eos,PetscViewer viewer){
    PetscFunctionBeginUser;

    PetscBool isAscii;
    PetscErrorCode ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII,  &isAscii);CHKERRQ(ierr);

    if (isAscii){
        ierr = PetscViewerASCIIPrintf(viewer, "EOS: %s\n", eos->type);CHKERRQ(ierr);
        if (eos->eosView) {
            ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
            ierr = eos->eosView(eos, viewer);CHKERRQ(ierr);
            ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode EOSDestroy(EOSData* eos) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    if ((*eos)->eosDestroy){
        (*eos)->eosDestroy(*eos);
    }

    PetscFree((*eos)->type);
    free(*eos);
    eos = NULL;
    PetscFunctionReturn(0);
}

PetscErrorCode EOSDecodeState(EOSData eos, const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* p){
    return eos->eosDecodeState(eos, yi, dim, density, totalEnergy, velocity, internalEnergy, a, p);
}

PetscErrorCode EOSTemperature(EOSData eos, const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, PetscReal* T) {
    return eos->eosTemperature(eos, yi, dim, density, totalEnergy, massFlux, T);
}

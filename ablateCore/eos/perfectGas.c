#include "perfectGas.h"

static PetscErrorCode EOSView_PerfectGas(EOSData eos, PetscViewer viewer){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    EOSData_PerfectGas perfectGasData = (EOSData_PerfectGas) eos->data;
    ierr = PetscViewerASCIIPrintf(viewer, "gamma: %f\n", perfectGasData->gamma);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Rgas: %f\n", perfectGasData->Rgas);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode EOSDestroy_PerfectGas(EOSData eos){
    PetscFunctionBeginUser;
    PetscErrorCode ierr = PetscFree((eos->data));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode EOSDecodeState_PerfectGas(EOSData eos, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* p){
    PetscFunctionBeginUser;
    PetscErrorCode ierr = PetscFree((eos->data));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode EOSSetFromOptions_PerfectGas(EOSData eos){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    EOSData_PerfectGas perfectGasData;
    PetscNew(&perfectGasData);
    eos->data =perfectGasData;

    // set default values for options
    perfectGasData->gamma = 1.4;
    perfectGasData->Rgas = 287.0;

    // get input parameters from the specified options database
    ierr = PetscOptionsGetReal(eos->options, NULL, "-gamma",&(perfectGasData->gamma), NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(eos->options, NULL, "-Rgas",&(perfectGasData->Rgas), NULL);CHKERRQ(ierr);

    // link the private methods
    eos->eosView = EOSView_PerfectGas;
    eos->eosDestroy = EOSDestroy_PerfectGas;
    eos->eosDecodeState = EOSDecodeState_PerfectGas;

    PetscFunctionReturn(0);
}


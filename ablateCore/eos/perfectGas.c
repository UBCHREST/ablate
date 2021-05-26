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

static PetscErrorCode EOSDecodeState_PerfectGas(EOSData eos, const PetscReal* yi,PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* p){
    PetscFunctionBeginUser;
    EOSData_PerfectGas perfectGasData = (EOSData_PerfectGas)eos->data;

    // Get the velocity in this direction
    PetscReal ke = 0.0;
    for (PetscInt d =0; d < dim; d++){
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;

    // assumed eos
    (*internalEnergy) = (totalEnergy) - ke;
    *p = (perfectGasData->gamma - 1.0)*density*(*internalEnergy);
    *a = PetscSqrtReal(perfectGasData->gamma*(*p)/density);
    PetscFunctionReturn(0);
}

static PetscErrorCode EOSTemperature_PerfectGas(EOSData eos, const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, PetscReal* T){
    PetscFunctionBeginUser;
    EOSData_PerfectGas perfectGasData = (EOSData_PerfectGas)eos->data;

    // Get the velocity in this direction
    PetscReal speedSquare = 0.0;
    for (PetscInt d =0; d < dim; d++){
        speedSquare += PetscSqr(massFlux[d]/density);
    }

    // assumed eos
    PetscReal internalEnergy = (totalEnergy) - 0.5 * speedSquare;
    PetscReal p = (perfectGasData->gamma - 1.0)*density*internalEnergy;

    (*T) = p/(perfectGasData->Rgas*density);
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
    eos->eosTemperature = EOSTemperature_PerfectGas;

    PetscFunctionReturn(0);
}


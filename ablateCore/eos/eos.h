#ifndef ABLATELIBRARY_EOS_H
#define ABLATELIBRARY_EOS_H
#include <petsc.h>

struct _EOSData{
    // the specific type of the eos
    char* type;

    // the options databased used to setup the options and children
    PetscOptions options;

    // implementation-specific data
    void* data;

    // implementation-specific methods
    PetscErrorCode (*eosView)(struct _EOSData* eos,PetscViewer viewer);
    PetscErrorCode (*eosDestroy)(struct _EOSData* eos);
    PetscErrorCode (*eosDecodeState)(struct _EOSData* eos, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* p);
};

typedef struct _EOSData* EOSData;

typedef PetscErrorCode (*EOSSetupFunction)(EOSData eos);

PETSC_EXTERN PetscErrorCode EOSRegister(const char*,const EOSSetupFunction);

// EOS object management methods
PETSC_EXTERN PetscErrorCode EOSCreate(EOSData* eos);
PETSC_EXTERN PetscErrorCode EOSSetType(EOSData eos, const char*);
PETSC_EXTERN PetscErrorCode EOSSetOptions(EOSData eos, PetscOptions options);
PETSC_EXTERN PetscErrorCode EOSSetFromOptions(EOSData eos);
PETSC_EXTERN PetscErrorCode EOSInitializeSpecies(EOSData eos, const char**);
PETSC_EXTERN PetscErrorCode EOSView(EOSData eos, PetscViewer viewer);
PETSC_EXTERN PetscErrorCode EOSDestroy(EOSData* eos);

// EOS use methods
/**
 * Support method for the eos that combines multiple decode values for efficiency
 * @param eos
 * @param dim
 * @param density
 * @param totalEnergy
 * @param velocity
 * @param[out] internalEnergy the internal energy of the flow
 * @param[out] a the speed of sound of the flow
 * @param[out] p pressure of the flow
 * @return
 */
PETSC_EXTERN PetscErrorCode EOSDecodeState(EOSData eos, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* p);

#endif  // ABLATELIBRARY_EOS_H

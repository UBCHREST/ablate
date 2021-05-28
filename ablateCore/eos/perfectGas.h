#ifndef ABLATELIBRARY_PERFECTGAS_H
#define ABLATELIBRARY_PERFECTGAS_H

#include "eos.h"

struct _EOSData_PerfectGas{
    PetscReal gamma;
    PetscReal Rgas;
};

typedef struct _EOSData_PerfectGas* EOSData_PerfectGas;

PetscErrorCode EOSSetFromOptions_PerfectGas(EOSData eos);

#endif  // ABLATELIBRARY_PERFECTGAS_H

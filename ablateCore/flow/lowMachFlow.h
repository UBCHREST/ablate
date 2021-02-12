#if !defined(lowMachFlow_h)
#define lowMachFlow_h
#include <petsc.h>
#include "flow.h"

// Define the functions for the lowMachFlow class
PETSC_EXTERN PetscErrorCode LowMachFlowCreate(Flow flow);

#endif
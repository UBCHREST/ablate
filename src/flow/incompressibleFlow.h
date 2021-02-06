#if !defined(incompressibleFlow_h)
#define lowMachFlow_h
#include <petsc.h>
#include "flow.h"


// Define the functions for the incompressibleFlow class
PETSC_EXTERN PetscErrorCode IncompressibleFlowCreate(Flow flow);

#endif
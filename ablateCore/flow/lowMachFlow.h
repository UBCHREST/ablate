#if !defined(lowMachFlow_h)
#define lowMachFlow_h
#include <petsc.h>
#include "flow.h"

// Define the test field number
#define VTEST 0
#define QTEST 1
#define WTEST 2

typedef enum { STROUHAL, REYNOLDS, FROUDE, PECLET, HEATRELEASE, GAMMA, PTH, MU, K, CP, BETA, GRAVITY_DIRECTION, TOTAL_LOW_MACH_FLOW_PARAMETERS } LowMachFlowParametersTypes;

typedef enum {VEL, PRES, TEMP, TOTAL_LOW_MACH_FLOW_FIELDS} LowMachFlowFields;

typedef enum {MOM, MASS, ENERGY, TOTAL_LOW_MACH_SOURCE_FIELDS} LowMachSourceFields;

PETSC_EXTERN PetscErrorCode FlowSetFromOptions_LowMachFlow(FlowData);

#endif
#if !defined(incompressibleFlow_h)
#define incompressibleFlow_h
#include <petsc.h>
#include "flow.h"

// Define the test field number
#define VTEST 0
#define QTEST 1
#define WTEST 2

typedef enum { STROUHAL, REYNOLDS, PECLET, MU, K, CP, TOTAL_INCOMPRESSIBLE_FLOW_PARAMETERS } IncompressibleFlowParametersTypes;

typedef enum {VEL, PRES, TEMP, TOTAL_INCOMPRESSIBLE_FLOW_FIELDS} IncompressibleFlowFields;

typedef enum {MOM, MASS, ENERGY, TOTAL_INCOMPRESSIBLE_SOURCE_FIELDS} IncompressibleSourceFields;

PetscErrorCode FlowSetFromOptions_IncompressibleFlow(FlowData flow);

#endif
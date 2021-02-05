#ifndef ABLATE_FLOW_H
#define ABLATE_FLOW_H

#include "../parameters.h"
#include <petsc.h>

// Define field id for velocity, pressure, temperature
#define VEL 0
#define PRES 1
#define TEMP 2

// Define the test field number
#define V 0
#define Q 1
#define W 2

typedef const char* FlowType;
#define FLOWLOWMACH         "flowLowMach"
#define FLOWINCOMPRESSIBLE         "flowIncompressible"

struct _Flow {
    PetscBag parameters; /* constant flow parameters */
    DM dm; /* flow domain */
    Vec flowField; /* flow solution vector */

    void *data;     /* implementation-specific data */

    /* flow field setup parameters */
    PetscErrorCode (*setupDiscretization)(struct _Flow*);
    PetscErrorCode (*startProblemSetup)(struct _Flow*);
    PetscErrorCode (*completeProblemSetup)(struct _Flow*, TS ts);
    PetscErrorCode (*completeFlowInitialization)(DM dm, Vec u);
    PetscErrorCode (*destroy)(struct _Flow*);
};

typedef struct _Flow* Flow;


PETSC_EXTERN PetscErrorCode FlowCreate(Flow* flow, FlowType type, DM dm);
PETSC_EXTERN PetscErrorCode FlowSetupDiscretization(Flow flow);
PETSC_EXTERN PetscErrorCode FlowStartProblemSetup(Flow flow);
PETSC_EXTERN PetscErrorCode FlowCompleteProblemSetup(Flow flow, TS ts);
PETSC_EXTERN PetscErrorCode FlowInitialization(Flow flow,DM dm, Vec u);
PETSC_EXTERN PetscErrorCode FlowDestroy(Flow* flow);




#endif  // ABLATE_FLOW_H

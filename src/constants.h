#if !defined(constants_h)
#include <petsc.h>

typedef enum {
    STROUHAL = 0,
    REYNOLDS,
    FROUDE,
    PECLET,
    HEATRELEASE,
    GAMMA,
    MU,
    K,
    CP,
    BETA,
    TOTAlCONSTANTS
} FlowConstants;

typedef struct {
    PetscReal strouhal;
    PetscReal reynolds;
    PetscReal froude;
    PetscReal peclet;
    PetscReal heatRelease;
    PetscReal gamma;
    PetscReal mu;    /* non-dimensional viscosity */
    PetscReal k;     /* non-dimensional thermal conductivity */
    PetscReal cp;    /* non-dimensional specific heat capacity */
    PetscReal beta;  /* non-dimensional thermal expansion coefficient */
} FlowParameters;

PETSC_EXTERN void PackFlowParameters(FlowParameters *parameters, PetscScalar *constantArray);

#endif
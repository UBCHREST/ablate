#if !defined(fvSupport_h)
#define fvSupport_h

#include <petsc.h>

PETSC_EXTERN PetscErrorCode DMPlexReconstructGradientsFVM_MulfiField(DM dm, PetscFV fvm,  Vec locX, Vec grad);
PETSC_EXTERN PetscErrorCode DMPlexGetDataFVM_MulfiField(DM dm, PetscFV fv, Vec *cellgeom, Vec *facegeom, DM *gradDM);

#endif
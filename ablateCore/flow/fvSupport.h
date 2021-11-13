#if !defined(fvSupport_h)
#define fvSupport_h

#include <petsc.h>

/**
 * reproduces the petsc call with grad fixes for multiple fields
 * @param dm
 * @param fvm
 * @param locX
 * @param grad
 * @return
 */
PETSC_EXTERN PetscErrorCode DMPlexGetDataFVM_MulfiField(DM dm, PetscFV fv, Vec *cellgeom, Vec *facegeom, DM *gradDM);

/**
 * Check to make sure local ghost boundary have valid gradient values
 */
PETSC_EXTERN PetscErrorCode ABLATE_FillGradientBoundary(DM dm, PetscFV auxFvm, Vec localXVec, Vec gradLocalVec);

#endif
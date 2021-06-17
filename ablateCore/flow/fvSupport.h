#if !defined(fvSupport_h)
#define fvSupport_h

#include <petsc.h>

/*@
  DMPlexTSComputeRHSFunctionFVM - Form the local forcing F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution
- user - The user context

  Output Parameter:
. F  - Global output vector

  Level: developer

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PETSC_EXTERN PetscErrorCode ABLATE_DMPlexTSComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *user);

/**
 * Takes all local vector
 * DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user
 * @return
 */
PETSC_EXTERN PetscErrorCode ABLATE_DMPlexComputeResidual_Internal(DM, IS, PetscReal, Vec, Vec, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode DMPlexReconstructGradientsFVM_MulfiField(DM dm, PetscFV fvm,  Vec locX, Vec grad);
PETSC_EXTERN PetscErrorCode DMPlexGetDataFVM_MulfiField(DM dm, PetscFV fv, Vec *cellgeom, Vec *facegeom, DM *gradDM);

#endif
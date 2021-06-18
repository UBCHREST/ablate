#if !defined(fvSupport_h)
#define fvSupport_h

#include <petsc.h>

#define MAX_FVM_RHS_FUNCTION_FIELDS 4

typedef PetscErrorCode (*FVMRHSFunction)(PetscInt dim, const PetscFVFaceGeom* fg, const PetscFVCellGeom* cgL, const PetscFVCellGeom* cgR,
                                          const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar* gradL[], const PetscScalar* gradR[],
                                          const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar* gradAuxL[], const PetscScalar* gradAuxR[],
                                              PetscScalar flux[], void* ctx);

struct FVMRHSFunctionDescription{
    FVMRHSFunction function;
    void* context;

    PetscInt field;
    PetscInt inputFields[MAX_FVM_RHS_FUNCTION_FIELDS];
    PetscInt numberInputFields;

    PetscInt auxFields[MAX_FVM_RHS_FUNCTION_FIELDS];
    PetscInt numberAuxFields;
};

typedef struct FVMRHSFunctionDescription FVMRHSFunctionDescription;

/**
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
**/
PETSC_EXTERN PetscErrorCode ABLATE_DMPlexTSComputeRHSFunctionFVM(FVMRHSFunctionDescription functionDescription[], PetscInt numberFunctionDescription, DM dm, PetscReal time, Vec locX, Vec F, void *user);

/**
 * Populate the boundary with gradient information
 * @param dm
 * @param auxDM
 * @param time
 * @param locXVec
 * @param locAuxField
 * @param numberUpdateFunctions
 * @param updateFunctions
 * @param data
 * @return
 */

/**
 * Takes all local vector
 * DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user
 * @return
 */
PETSC_EXTERN PetscErrorCode ABLATE_DMPlexComputeResidual_Internal(FVMRHSFunctionDescription functionDescription[], PetscInt numberFunctionDescription, DM, IS, PetscReal, Vec, Vec, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode DMPlexReconstructGradientsFVM_MulfiField(DM dm, PetscFV fvm,  Vec locX, Vec grad);
PETSC_EXTERN PetscErrorCode DMPlexGetDataFVM_MulfiField(DM dm, PetscFV fv, Vec *cellgeom, Vec *facegeom, DM *gradDM);

#endif
#if !defined(fvSupport_h)
#define fvSupport_h

#include <petsc.h>

#define MAX_FVM_RHS_FUNCTION_FIELDS 4

typedef PetscErrorCode (*FVMRHSFluxFunction)(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                             const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[],
                                             const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar flux[], void *ctx);

typedef PetscErrorCode (*FVMRHSPointFunction)(PetscInt dim, const PetscFVCellGeom *cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void *ctx);

typedef PetscErrorCode (*FVAuxFieldUpdateFunction)(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx);

/**
 * struct to describe how to compute RHS finite volume flux source terms
 */
struct _FVMRHSFluxFunctionDescription {
    FVMRHSFluxFunction function;
    void *context;

    PetscInt field;
    PetscInt inputFields[MAX_FVM_RHS_FUNCTION_FIELDS];
    PetscInt numberInputFields;

    PetscInt auxFields[MAX_FVM_RHS_FUNCTION_FIELDS];
    PetscInt numberAuxFields;
};

typedef struct _FVMRHSFluxFunctionDescription FVMRHSFluxFunctionDescription;

/**
 * struct to describe how to compute RHS finite volume point source terms
 */
struct _FVMRHSPointFunctionDescription {
    FVMRHSPointFunction function;
    void *context;

    PetscInt fields[MAX_FVM_RHS_FUNCTION_FIELDS];
    PetscInt numberFields;

    PetscInt inputFields[MAX_FVM_RHS_FUNCTION_FIELDS];
    PetscInt numberInputFields;

    PetscInt auxFields[MAX_FVM_RHS_FUNCTION_FIELDS];
    PetscInt numberAuxFields;
};

typedef struct _FVMRHSPointFunctionDescription FVMRHSPointFunctionDescription;

/**
 * struct to describe the point Jacobin functions
 */
struct _FVMRHSPointJacobianDescription {
    FVMRHSPointFunction function;
    void *context;

    PetscInt fields[MAX_FVM_RHS_FUNCTION_FIELDS];
    PetscInt numberFields;
};

typedef struct _FVMRHSPointJacobianDescription FVMRHSPointJacobianDescription;

/**
  DMPlexTSComputeRHSFunctionFVM - Form the local forcing F from the local input X using flux and pointfunctions specified by the user

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
PETSC_EXTERN PetscErrorCode ABLATE_DMPlexComputeRHSFunctionFVM(FVMRHSFluxFunctionDescription *fluxFunctionDescriptions, PetscInt numberFluxFunctionDescription,
                                                               FVMRHSPointFunctionDescription *pointFunctionDescriptions, PetscInt numberPointFunctionDescription,
                                                               DM dm, PetscReal time, Vec locX, Vec F);

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
 * @return
 */
PETSC_EXTERN PetscErrorCode ABLATE_DMPlexComputeFluxResidual_Internal(FVMRHSFluxFunctionDescription functionDescription[], PetscInt numberFunctionDescription, DM, IS, PetscReal, Vec, Vec, PetscReal, Vec);

/**
   Form the local forcing F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. t - The time
. locX  - Local solution

  Output Parameter:
. F  - local output vector

**/
PETSC_EXTERN PetscErrorCode ABLATE_DMPlexComputePointResidual_Internal(FVMRHSPointFunctionDescription *functionDescription, PetscInt numberFunctionDescription, DM, IS, PetscReal, Vec, Vec, PetscReal, Vec);

/**
   Form the point wise jacobian for the FVM method
**/
PETSC_EXTERN PetscErrorCode ABLATE_DMPlexComputeRHSJacobianFVM(FVMRHSPointJacobianDescription *functionDescription, PetscInt numberFunctionDescription,  DM dm, PetscReal t,Vec u,Mat aMat,Mat pMat);

/**
 * reproduces the petsc call with grad fixes for multiple fields
 * @param dm
 * @param fvm
 * @param locX
 * @param grad
 * @return
 */
PETSC_EXTERN PetscErrorCode DMPlexReconstructGradientsFVM_MulfiField(DM dm, PetscFV fvm, Vec locX, Vec grad);

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
 * Function to update all cells.  This should be merged into other update calls
 * @param dm
 * @param auxDM
 * @param time
 * @param locXVec
 * @param locAuxField
 * @param numberUpdateFunctions
 * @param updateFunctions
 * @param ctx
 * @return
 */
PETSC_EXTERN PetscErrorCode FVFlowUpdateAuxFieldsFV(DM dm, DM auxDM, PetscReal time, Vec locXVec, Vec locAuxField, PetscInt numberUpdateFunctions, FVAuxFieldUpdateFunction* updateFunctions, void** ctx);



#endif
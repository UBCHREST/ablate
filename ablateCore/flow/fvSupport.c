#include "fvSupport.h"
#include <inttypes.h>

/**
 * Internal petsc function that is required.  The exported function DMPlexReconstructGradients does not allow using any fvm or grad when nFields > 0
 * @param dm
 * @param fvm
 * @param fStart
 * @param fEnd
 * @param faceGeometry
 * @param cellGeometry
 * @param locX
 * @param grad
 * @return
 */
PetscErrorCode DMPlexReconstructGradients_Internal(DM dm, PetscFV fvm, PetscInt fStart, PetscInt fEnd, Vec faceGeometry, Vec cellGeometry, Vec locX, Vec grad);

/*@
  DMPlexReconstructGradientsFVM - reconstruct the gradient of a vector using a finite volume method for a specific field

  Input Parameters:
+ dm - the mesh
+ fvm - the fvm
- locX - the local representation of the vector

  Output Parameter:
. grad - the global representation of the gradient

  Level: developer

.seealso: DMPlexGetGradientDM()
@*/
PetscErrorCode DMPlexReconstructGradientsFVM_MulfiField(DM dm, PetscFV fvm,  Vec locX, Vec grad)
{
    PetscDS          prob;
    PetscInt         fStart, fEnd;
    Vec              faceGeometryFVM, cellGeometryFVM;
    PetscFVCellGeom  *cgeomFVM   = NULL;
    PetscFVFaceGeom  *fgeomFVM   = NULL;
    DM               dmGrad = NULL;
    PetscErrorCode   ierr;

    PetscFunctionBegin;
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);

    ierr = DMPlexGetDataFVM_MulfiField(dm, fvm, &cellGeometryFVM, &faceGeometryFVM, &dmGrad);CHKERRQ(ierr);
    if (!dmGrad) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"This dm's finite volume discretization does not reconstruct gradients");
    ierr = VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


/*@
  DMPlexGetDataFVM - Retrieve precomputed cell geometry

  Collective on dm

  Input Arguments:
+ dm  - The DM
- fvm - The PetscFV

  Output Parameters:
+ cellGeometry - The cell geometry
. faceGeometry - The face geometry
- dmGrad       - The gradient matrices

  Level: developer

.seealso: DMPlexComputeGeometryFVM()
@*/
PetscErrorCode DMPlexGetDataFVM_MulfiField(DM dm, PetscFV fv, Vec *cellgeom, Vec *facegeom, DM *gradDM)
{
    PetscObject    cellgeomobj, facegeomobj;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_cellgeom_fvm", &cellgeomobj);CHKERRQ(ierr);
    if (!cellgeomobj) {
        Vec cellgeomInt, facegeomInt;

        ierr = DMPlexComputeGeometryFVM(dm, &cellgeomInt, &facegeomInt);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject) dm, "DMPlex_cellgeom_fvm",(PetscObject)cellgeomInt);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject) dm, "DMPlex_facegeom_fvm",(PetscObject)facegeomInt);CHKERRQ(ierr);
        ierr = VecDestroy(&cellgeomInt);CHKERRQ(ierr);
        ierr = VecDestroy(&facegeomInt);CHKERRQ(ierr);
        ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_cellgeom_fvm", &cellgeomobj);CHKERRQ(ierr);
    }
    ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_facegeom_fvm", &facegeomobj);CHKERRQ(ierr);
    if (cellgeom) *cellgeom = (Vec) cellgeomobj;
    if (facegeom) *facegeom = (Vec) facegeomobj;
    if (gradDM) {
        PetscObject gradobj;
        PetscBool   computeGradients;

        ierr = PetscFVGetComputeGradients(fv,&computeGradients);CHKERRQ(ierr);
        if (!computeGradients) {
            *gradDM = NULL;
            PetscFunctionReturn(0);
        }

        // Get the petscId object for this fv
        PetscObjectId fvId;
        ierr = PetscObjectGetId((PetscObject) fv, &fvId);CHKERRQ(ierr);

        char dmGradName[PETSC_MAX_OPTION_NAME];
        ierr = PetscSNPrintf(dmGradName, PETSC_MAX_OPTION_NAME, "DMPlex_dmgrad_fvm_%" PRId64,  fvId);CHKERRQ(ierr);

        ierr = PetscObjectQuery((PetscObject) dm, dmGradName, &gradobj);CHKERRQ(ierr);
        if (!gradobj) {
            DM dmGradInt;

            ierr = DMPlexComputeGradientFVM(dm,fv,(Vec) facegeomobj,(Vec) cellgeomobj,&dmGradInt);CHKERRQ(ierr);
            ierr = PetscObjectCompose((PetscObject) dm, dmGradName, (PetscObject)dmGradInt);CHKERRQ(ierr);
            ierr = DMDestroy(&dmGradInt);CHKERRQ(ierr);
            ierr = PetscObjectQuery((PetscObject) dm, dmGradName, &gradobj);CHKERRQ(ierr);
        }
        *gradDM = (DM) gradobj;
    }
    PetscFunctionReturn(0);
}

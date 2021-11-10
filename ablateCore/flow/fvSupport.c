#include "fvSupport.h"
#include <inttypes.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/petscfvimpl.h> /*I "petscfv.h" I*/

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




/**
 * Hard coded function to compute the boundary cell gradient.  This should be relaxed to a boundary condition
 * @param dim
 * @param dof
 * @param faceGeom
 * @param cellGeom
 * @param cellGeomG
 * @param a_xI
 * @param a_xGradI
 * @param a_xG
 * @param a_xGradG
 * @param ctx
 * @return
 */
static PetscErrorCode ComputeBoundaryCellGradient(PetscInt dim, PetscInt dof, const PetscFVFaceGeom *faceGeom, const PetscFVCellGeom *cellGeom, const PetscFVCellGeom *cellGeomG, const PetscScalar *a_xI, const PetscScalar *a_xGradI, const PetscScalar *a_xG,  PetscScalar *a_xGradG, void *ctx){
    PetscFunctionBeginUser;

    for (PetscInt pd = 0; pd < dof; ++pd) {
        PetscReal dPhidS = a_xG[pd] - a_xI[pd];

        // over each direction
        for (PetscInt dir = 0; dir < dim; dir++) {
            PetscReal dx = (cellGeomG->centroid[dir] - cellGeom->centroid[dir])/2.0;

            // If there is a contribution in this direction
            if (PetscAbs(dx) > 1E-8) {
                a_xGradG[pd*dim + dir] = dPhidS / dx;
            } else {
                a_xGradG[pd*dim + dir] = 0.0;
            }
        }
    }
    PetscFunctionReturn(0);
}


/**
 * this function updates the boundaries with the gradient computed from the boundary cell value
 * @param dm
 * @param auxFvm
 * @param gradLocalVec
 * @return
 */
PetscErrorCode ABLATE_FillGradientBoundary(DM dm, PetscFV auxFvm, Vec localXVec, Vec gradLocalVec) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Get the dmGrad
    DM dmGrad;
    ierr = VecGetDM(gradLocalVec, &dmGrad);CHKERRQ(ierr);

    // get the problem
    PetscDS prob;
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);

    PetscInt field;
    ierr = PetscDSGetFieldIndex(prob, (PetscObject) auxFvm, &field);CHKERRQ(ierr);
    PetscInt dof;
    ierr = PetscDSGetFieldSize(prob, field, &dof);CHKERRQ(ierr);

    // Obtaining local cell ownership
    PetscInt faceStart, faceEnd;
    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);CHKERRQ(ierr);

    // Get the fvm face and cell geometry
    Vec cellGeomVec = NULL;/* vector of structs related to cell geometry*/
    Vec faceGeomVec = NULL;/* vector of structs related to face geometry*/
    ierr = DMPlexGetGeometryFVM(dm, &faceGeomVec, &cellGeomVec, NULL);CHKERRQ(ierr);

    // get the dm for each geom type
    DM dmFaceGeom, dmCellGeom;
    ierr = VecGetDM(faceGeomVec, &dmFaceGeom);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCellGeom);CHKERRQ(ierr);

    // extract the gradLocalVec
    PetscScalar * gradLocalArray;
    ierr = VecGetArray(gradLocalVec, &gradLocalArray);CHKERRQ(ierr);

    const PetscScalar* localArray;
    ierr = VecGetArrayRead(localXVec, &localArray);CHKERRQ(ierr);

    // get the dim
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // extract the arrays for the face and cell geom, along with their dm
    const PetscScalar *cellGeomArray;
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);

    const PetscScalar *faceGeomArray;
    ierr = VecGetArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);

    // march over each boundary for this problem
    PetscInt numBd;
    ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
    for (PetscInt b = 0; b < numBd; ++b) {
        // extract the boundary information
        PetscInt                numids;
        const PetscInt         *ids;
        DMLabel                 label;
        const char * name;
        PetscInt boundaryField;
        ierr = PetscDSGetBoundary(prob, b, NULL, NULL, &name, &label, &numids, &ids, &boundaryField, NULL, NULL, NULL, NULL,  NULL);CHKERRQ(ierr);

        if (boundaryField != field){
            continue;
        }

        // get the correct boundary/ghost pattern
        PetscSF            sf;
        PetscInt        nleaves;
        const PetscInt    *leaves;
        ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
        ierr = PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL);CHKERRQ(ierr);
        nleaves = PetscMax(0, nleaves);

        // march over each id on this process
        for (PetscInt i = 0; i < numids; ++i) {
            IS              faceIS;
            const PetscInt *faces;
            PetscInt        numFaces, f;

            ierr = DMLabelGetStratumIS(label, ids[i], &faceIS);CHKERRQ(ierr);
            if (!faceIS) continue; /* No points with that id on this process */
            ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
            ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);

            // march over each face in this boundary
            for (f = 0; f < numFaces; ++f) {
                const PetscInt* cells;

                if ((faces[f] < faceStart) || (faces[f] >= faceEnd)){
                    continue; /* Refinement adds non-faces to labels */
                }
                PetscInt loc;
                ierr = PetscFindInt(faces[f], nleaves, (PetscInt *) leaves, &loc);CHKERRQ(ierr);
                if (loc >= 0){
                    continue;
                }

                PetscBool boundary;
                ierr = DMIsBoundaryPoint(dm, faces[f], &boundary);CHKERRQ(ierr);

                // get the ghost and interior nodes
                ierr = DMPlexGetSupport(dm, faces[f], &cells);CHKERRQ(ierr);
                const PetscInt cellI = cells[0];
                const PetscInt cellG = cells[1];

                // get the face geom
                const PetscFVFaceGeom  *faceGeom;
                ierr  = DMPlexPointLocalRead(dmFaceGeom, faces[f], faceGeomArray, &faceGeom);CHKERRQ(ierr);

                // get the cell centroid information
                const PetscFVCellGeom       *cellGeom;
                const PetscFVCellGeom       *cellGeomGhost;
                ierr  = DMPlexPointLocalRead(dmCellGeom, cellI, cellGeomArray, &cellGeom);CHKERRQ(ierr);
                ierr  = DMPlexPointLocalRead(dmCellGeom, cellG, cellGeomArray, &cellGeomGhost);CHKERRQ(ierr);

                // Read the local point
                PetscScalar* boundaryGradCellValues;
                ierr  = DMPlexPointLocalRef(dmGrad, cellG, gradLocalArray, &boundaryGradCellValues);CHKERRQ(ierr);

                const PetscScalar*  cellGradValues;
                ierr  = DMPlexPointLocalRead(dmGrad, cellI, gradLocalArray, &cellGradValues);CHKERRQ(ierr);

                const PetscScalar* boundaryCellValues;
                ierr  = DMPlexPointLocalFieldRead(dm, cellG, field, localArray, &boundaryCellValues);CHKERRQ(ierr);
                const PetscScalar* cellValues;
                ierr  = DMPlexPointLocalFieldRead(dm, cellI, field, localArray, &cellValues);CHKERRQ(ierr);

                // compute the gradient for the boundary node and pass in
                ierr = ComputeBoundaryCellGradient(dim, dof, faceGeom, cellGeom, cellGeomGhost, cellValues, cellGradValues, boundaryCellValues, boundaryGradCellValues, NULL);CHKERRQ(ierr);
            }
            ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
            ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
        }
    }

    ierr = VecRestoreArrayRead(localXVec, &localArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(gradLocalVec, &gradLocalArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
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

PetscErrorCode ABLATE_DMPlexComputeRHSFunctionFVM(FVMRHSFluxFunctionDescription *fluxFunctionDescription, PetscInt numberFluxFunctionDescription,
                                                      FVMRHSPointFunctionDescription *pointFunctionDescriptions, PetscInt numberPointFunctionDescription,
                                                      DM dm, PetscReal time, Vec locX, Vec F)
{
    Vec            locF;
    IS             cellIS;
    DM             plex;
    PetscInt       depth;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = DMConvert(dm,DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
    ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
    if (!cellIS) {
        ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);
    }
    ierr = DMGetLocalVector(plex, &locF);CHKERRQ(ierr);
    ierr = VecZeroEntries(locF);CHKERRQ(ierr);

    // compute the contribution from fluxes
    ierr = ABLATE_DMPlexComputeFluxResidual_Internal(fluxFunctionDescription, numberFluxFunctionDescription, plex, cellIS, time, locX, NULL, time, locF);CHKERRQ(ierr);

    // compute the contribution from point sources
    ierr = ABLATE_DMPlexComputePointResidual_Internal(pointFunctionDescriptions, numberPointFunctionDescription, plex, cellIS, time, locX, NULL, time, locF);CHKERRQ(ierr);

    ierr = DMLocalToGlobalBegin(plex, locF, ADD_VALUES, F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(plex, locF, ADD_VALUES, F);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(plex, &locF);CHKERRQ(ierr);
    ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


/*@C
  DMPlexGetFaceFields - Retrieve the field values values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
. faceGeometry - A local vector with face geometry
. cellGeometry - A local vector with cell geometry
- locaGrad - A local vector with field gradients, or NULL

  Output Parameters:
+ Nface - The number of faces with field values
. uL - The field values at the left side of the face
- uR - The field values at the right side of the face
- gradL - The grad field values at the left side fo the face
- gradR - The grad field values on the right side of the face
  Level: developer

.seealso: DMPlexGetCellFields()
@*/
static PetscErrorCode ABLATE_DMPlexGetFaceFields(DM dm, PetscInt fStart, PetscInt fEnd, Vec locX, Vec faceGeometry, Vec cellGeometry, const Vec* locGrads, PetscInt *Nface, PetscScalar **uL, PetscScalar **uR, PetscScalar **gradL, PetscScalar **gradR, PetscBool projectField)
{
    DM                 dmFace, dmCell, *dmGrads;
    PetscSection       section;
    PetscDS            prob;
    DMLabel            ghostLabel;
    const PetscScalar *facegeom, *cellgeom, *x, **lgrads;
    PetscBool         *isFE;
    PetscInt           dim, Nf, f, Nc, numFaces = fEnd - fStart, iface, face;
    PetscErrorCode     ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
    PetscValidHeaderSpecific(locX, VEC_CLASSID, 4);
    PetscValidHeaderSpecific(faceGeometry, VEC_CLASSID, 6);
    PetscValidHeaderSpecific(cellGeometry, VEC_CLASSID, 7);
    PetscValidPointer(uL, 9);
    PetscValidPointer(uR, 10);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
    ierr = PetscDSGetTotalComponents(prob, &Nc);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nf, &isFE);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;

        ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
        else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
        else                            {isFE[f] = PETSC_FALSE;}
    }
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locX, &x);CHKERRQ(ierr);
    ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, numFaces*Nc, MPIU_SCALAR, uL);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, numFaces*Nc, MPIU_SCALAR, uR);CHKERRQ(ierr);

    if (locGrads) {
        ierr = PetscCalloc1(Nf, &lgrads);CHKERRQ(ierr);
        ierr = PetscCalloc1(Nf, &dmGrads);CHKERRQ(ierr);
        for (f = 0; f < Nf; ++f) {
            if (locGrads[f]) {
                ierr = VecGetArrayRead(locGrads[f], &lgrads[f]);CHKERRQ(ierr);
                ierr = VecGetDM(locGrads[f], &dmGrads[f]);CHKERRQ(ierr);
            }
        }
        // size up the work arrays
        ierr = DMGetWorkArray(dm, dim*numFaces*Nc, MPIU_SCALAR, gradL);CHKERRQ(ierr);
        ierr = DMGetWorkArray(dm, dim*numFaces*Nc, MPIU_SCALAR, gradR);CHKERRQ(ierr);
    }else{
        *gradL = NULL;
        *gradR = NULL;
    }

    /* Right now just eat the extra work for FE (could make a cell loop) */
    for (face = fStart, iface = 0; face < fEnd; ++face) {
        const PetscInt *cells;
        PetscFVFaceGeom *fg;
        PetscFVCellGeom *cgL, *cgR;
        PetscScalar *xL, *xR, *gL, *gR;
        PetscScalar *uLl = *uL, *uRl = *uR;
        PetscScalar *gradLl = *gradL, *gradRl = *gradR;
        PetscInt ghost, nsupp, nchild;

        ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
        ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
        ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);

        // Keep track of derivative offset
        PetscInt *offsets;
        PetscInt *dirOffsets;
        ierr = PetscDSGetComponentOffsets(prob, &offsets);CHKERRQ(ierr);
        ierr = PetscDSGetComponentDerivativeOffsets(prob, &dirOffsets);CHKERRQ(ierr);

        // march over each field
        for (f = 0; f < Nf; ++f) {
            PetscFV fv;
            PetscInt numComp, c;

            ierr = PetscDSGetDiscretization(prob, f, (PetscObject *)&fv);CHKERRQ(ierr);
            ierr = PetscFVGetNumComponents(fv, &numComp);CHKERRQ(ierr);
            ierr = DMPlexPointLocalFieldRead(dm, cells[0], f, x, &xL);CHKERRQ(ierr);
            ierr = DMPlexPointLocalFieldRead(dm, cells[1], f, x, &xR);CHKERRQ(ierr);
            if (dmGrads[f] && projectField) {
                PetscReal dxL[3], dxR[3];

                ierr = DMPlexPointLocalRead(dmGrads[f], cells[0], lgrads[f], &gL);CHKERRQ(ierr);
                ierr = DMPlexPointLocalRead(dmGrads[f], cells[1], lgrads[f], &gR);CHKERRQ(ierr);
                DMPlex_WaxpyD_Internal(dim, -1, cgL->centroid, fg->centroid, dxL);
                DMPlex_WaxpyD_Internal(dim, -1, cgR->centroid, fg->centroid, dxR);
                // Project the cell centered value onto the face
                for (c = 0; c < numComp; ++c) {
                    uLl[iface * Nc + offsets[f] + c] = xL[c] + DMPlex_DotD_Internal(dim, &gL[c * dim], dxL);
                    uRl[iface * Nc + offsets[f] + c] = xR[c] + DMPlex_DotD_Internal(dim, &gR[c * dim], dxR);

                    // copy the gradient into the grad vector
                    for (PetscInt d = 0; d < dim; d++) {
                        gradLl[iface * Nc * dim + dirOffsets[f] + c * dim + d] = gL[c * dim + d];
                        gradRl[iface * Nc * dim + dirOffsets[f] + c * dim + d] = gR[c * dim + d];
                    }
                }
            } else if (dmGrads[f]) {
                PetscReal dxL[3], dxR[3];

                ierr = DMPlexPointLocalRead(dmGrads[f], cells[0], lgrads[f], &gL);CHKERRQ(ierr);
                ierr = DMPlexPointLocalRead(dmGrads[f], cells[1], lgrads[f], &gR);CHKERRQ(ierr);
                // Project the cell centered value onto the face
                for (c = 0; c < numComp; ++c) {
                    uLl[iface * Nc + offsets[f] + c] = xL[c];
                    uRl[iface * Nc + offsets[f] + c] = xR[c];

                    // copy the gradient into the grad vector
                    for (PetscInt d = 0; d < dim; d++) {
                        gradLl[iface * Nc * dim + dirOffsets[f] + c * dim + d] = gL[c * dim + d];
                        gradRl[iface * Nc * dim + dirOffsets[f] + c * dim + d] = gR[c * dim + d];
                    }
                }

            } else {
                // Just copy the cell centered value on to the face
                for (c = 0; c < numComp; ++c) {
                    uLl[iface * Nc + offsets[f] + c] = xL[c];
                    uRl[iface * Nc + offsets[f] + c] = xR[c];

                    // fill the grad with NAN to prevent use
                    for (PetscInt d = 0; d < dim; d++) {
                        gradLl[iface * Nc * dim + dirOffsets[f] + c * dim + d] = NAN;
                        gradRl[iface * Nc * dim + dirOffsets[f] + c * dim + d] = NAN;
                    }
                }
            }
        }
        ++iface;
    }
    *Nface = iface;
    ierr = VecRestoreArrayRead(locX, &x);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
    if (locGrads) {
            for (f = 0; f < Nf; ++f) {
                if (locGrads[f]) {
                    ierr = VecRestoreArrayRead(locGrads[f], &lgrads[f]);CHKERRQ(ierr);
                }
            }
        ierr = PetscFree(lgrads);CHKERRQ(ierr);
    }
    ierr = PetscFree(isFE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode ABLATE_DMPlexRestoreFaceFields(DM dm, PetscInt fStart, PetscInt fEnd, Vec locX, Vec locX_t, Vec faceGeometry, Vec cellGeometry, Vec locGrad, PetscInt *Nface, PetscScalar **uL, PetscScalar **uR,  PetscScalar **gradL, PetscScalar **gradR)
{

  DMRestoreWorkArray(dm, 0, MPIU_SCALAR, uL);
  DMRestoreWorkArray(dm, 0, MPIU_SCALAR, uR);
  if (*gradL){
      DMRestoreWorkArray(dm, 0, MPIU_SCALAR, gradL);
  }
  if (*gradR){
      DMRestoreWorkArray(dm, 0, MPIU_SCALAR, gradR);
  }

  return(0);
}



/*
 * Private function to compute the rhs based upon a FVMRHSFunctionDescription.
 *
 * Not it is assumed that the fvm object's field is the same one as in functionDescription
  neighborVol[f*2+0] contains the left  geom
  neighborVol[f*2+1] contains the right geom
*/
static PetscErrorCode ABLATE_PetscFVIntegrateRHSFunction(FVMRHSFluxFunctionDescription * functionDescription, PetscFV fvm, PetscDS prob, PetscDS auxProb, PetscInt numberFaces, PetscFVFaceGeom *fgeom, PetscReal *neighborVol,
                                                         PetscScalar uL[], PetscScalar uR[], PetscScalar gradL[], PetscScalar gradR[],
                                                         PetscScalar auxL[], PetscScalar auxR[], PetscScalar gradAuxL[], PetscScalar gradAuxR[],
                                                         PetscScalar fluxL[], PetscScalar fluxR[])
{
    void              *rctx;
    PetscScalar       *flux = fvm->fluxWork;
    PetscErrorCode     ierr;

    PetscFunctionBegin;

    // Get the total number of components (in all fields)
    PetscInt nCompTot;
    ierr = PetscDSGetTotalComponents(prob, &nCompTot);CHKERRQ(ierr);
    PetscInt nAuxCompTot;
    ierr = PetscDSGetTotalComponents(auxProb, &nAuxCompTot);CHKERRQ(ierr);
    PetscInt totalDim;//This is usually the same?
    ierr = PetscDSGetTotalDimension(prob, &totalDim);CHKERRQ(ierr);

    // create the required offset arrays
    PetscInt *uOff, *aOff, *uOff_x = NULL, *aOff_x = NULL;
    PetscCalloc1(functionDescription->numberInputFields, &uOff);
    PetscCalloc1(functionDescription->numberAuxFields, &aOff);
    PetscCalloc1(functionDescription->numberInputFields, &uOff_x);
    PetscCalloc1(functionDescription->numberAuxFields, &aOff_x);

    // Get the full set of offsets from the ds
    PetscInt * uOffTotal;
    PetscInt * uGradOffTotal;
    ierr = PetscDSGetComponentOffsets(prob, &uOffTotal);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(prob, &uGradOffTotal);CHKERRQ(ierr);
    for (PetscInt f =0; f < functionDescription->numberInputFields; f++){
        uOff[f] = uOffTotal[functionDescription->inputFields[f]];
        uOff_x[f] = uGradOffTotal[functionDescription->inputFields[f]];
    }

    if (auxProb) {
        PetscInt *auxOffTotal;
        PetscInt *auxGradOffTotal;
        ierr = PetscDSGetComponentOffsets(auxProb, &auxOffTotal);CHKERRQ(ierr);
        ierr = PetscDSGetComponentDerivativeOffsets(auxProb, &auxGradOffTotal);CHKERRQ(ierr);
        for (PetscInt f = 0; f < functionDescription->numberAuxFields; f++) {
            aOff[f] = auxOffTotal[functionDescription->auxFields[f]];
            aOff_x[f] = auxGradOffTotal[functionDescription->auxFields[f]];
        }
    }
    // get the flux offset from the field
    PetscInt fluxOffset;
    ierr = PetscDSGetFieldOffset(prob, functionDescription->field, &fluxOffset);CHKERRQ(ierr);

    PetscInt dim;
    ierr = PetscFVGetSpatialDimension(fvm, &dim);CHKERRQ(ierr);
    PetscInt fluxDim;
    ierr = PetscFVGetNumComponents(fvm, &fluxDim);CHKERRQ(ierr);
    // for each face, compute and copy
    for (PetscInt f = 0; f < numberFaces; ++f) {
        ierr = functionDescription->function(dim, &fgeom[f],
                              uOff, uOff_x, &uL[f*nCompTot], &uR[f*nCompTot], &gradL[f*nCompTot*dim], &gradR[f*nCompTot*dim],
                              aOff, aOff_x, &auxL[f*nAuxCompTot], &auxR[f*nAuxCompTot], &gradAuxL[f*nAuxCompTot*dim], &gradAuxR[f*nAuxCompTot*dim],
                              flux, functionDescription->context);CHKERRQ(ierr);
        for (PetscInt d = 0; d < fluxDim; ++d) {
            fluxL[f*totalDim+fluxOffset+d] += flux[d] / neighborVol[f*2+0];
            fluxR[f*totalDim+fluxOffset+d] += flux[d] / neighborVol[f*2+1];
        }
    }

    PetscFree(uOff);
    PetscFree(aOff);
    PetscFree(uOff_x);
    PetscFree(aOff_x);
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
            PetscReal dx = (cellGeomG->centroid[dir] - faceGeom->centroid[dir]);

            // If there is a contribution in this direction
            if (PetscAbs(dx) > 1E-8) {
                a_xGradG[pd*dim + dir] = dPhidS / (dx);
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
static PetscErrorCode ABLATE_FillGradientBoundary(DM dm, PetscFV auxFvm, Vec localXVec, Vec gradLocalVec){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Get the dmGrad
    DM dmGrad;
    ierr = VecGetDM(gradLocalVec, &dmGrad);CHKERRQ(ierr);

    // get the problem
    PetscDS prob;
    PetscInt nFields;
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
        const char *                 labelName;
        PetscInt boundaryField;
        ierr = DMGetBoundary(dm, b, NULL, NULL, &labelName, &boundaryField, NULL, NULL, NULL, NULL, &numids, &ids, NULL);CHKERRQ(ierr);

        if (boundaryField != field){
            continue;
        }

        // use the correct label for this boundary field
        DMLabel label;
        ierr = DMGetLabel(dm, labelName, &label);CHKERRQ(ierr);

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
                PetscFVFaceGeom        *fg;

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

PetscErrorCode ABLATE_DMPlexComputeFluxResidual_Internal(FVMRHSFluxFunctionDescription functionDescriptions[], PetscInt numberFunctionDescriptions, DM dm, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF)
{
    DM_Plex         *mesh       = (DM_Plex *) dm->data;
    const char      *name       = "Residual";
    DM               dmAux      = NULL;
    DM               *dmGrads, *dmAuxGrads    = NULL;
    DMLabel          ghostLabel = NULL;
    PetscDS          ds         = NULL;
    PetscDS          dsAux      = NULL;
    PetscSection     section    = NULL;
    PetscBool        isImplicit = (locX_t || time == PETSC_MIN_REAL) ? PETSC_TRUE : PETSC_FALSE;
    PetscFVCellGeom *cgeomFVM   = NULL;
    PetscFVFaceGeom *fgeomFVM   = NULL;
    DMField          coordField = NULL;
    Vec *locGrads, *locAuxGrads =NULL;  // each field will have a separate local gradient vector
    Vec              locA, cellGeometryFVM = NULL, faceGeometryFVM = NULL;
    PetscScalar     *u = NULL, *u_t, *a;
    PetscScalar     *uL, *uR, *gradL, *gradR;
    PetscScalar     *auxL, *auxR, *gradAuxL, *gradAuxR;
    IS               chunkIS;
    const PetscInt  *cells;
    PetscInt         cStart, cEnd, numCells;
    PetscInt nf, naf, totDim, totDimAux, numChunks, cellChunkSize, faceChunkSize, chunk, fStart, fEnd;
    PetscInt         maxDegree = PETSC_MAX_INT;
    PetscQuadrature  affineQuad = NULL, *quads = NULL;
    PetscFEGeom     *affineGeom = NULL, **geoms = NULL;
    PetscErrorCode   ierr;

    PetscFunctionBeginUser;
    /* FEM+FVM */
    ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    /* 1: Get sizes from dm and dmAux */
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
    ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(ds, &nf);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
    if (locA) {
        PetscInt subcell;
        ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
        ierr = DMGetEnclosurePoint(dmAux, dm, DM_ENC_UNKNOWN, cStart, &subcell);CHKERRQ(ierr);
        ierr = DMGetCellDS(dmAux, subcell, &dsAux);CHKERRQ(ierr);
        ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
        ierr = PetscDSGetNumFields(dsAux, &naf);CHKERRQ(ierr);
    }

    /* 2: Get geometric data */
    // We can use a single call for the geometry data because it does not depend on the fv object
    ierr = DMPlexGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);

    // Get the dm grad for each field
    ierr = PetscCalloc1(nf, &dmGrads);CHKERRQ(ierr);
    ierr = PetscCalloc1(nf, &locGrads);CHKERRQ(ierr);

    /* Reconstruct and limit cell gradients */
    // for each field compute the gradient in the localGrads vector
    for (PetscInt f = 0; f < nf; f++){
        PetscFV fvm;
        ierr = DMGetField(dm, f, NULL, (PetscObject*)&fvm);CHKERRQ(ierr);

        // Get the needed auxDm
        //ierr = DMPlexGetGradientDM(dm, fvm, &dmGrad);CHKERRQ(ierr);
        // this call replaces DMPlexGetGradientDM.  DMPlexGetGradientDM only grabs the first created dm, while this creates one (correct size) for each field
        ierr = DMPlexGetDataFVM_MulfiField(dm, fvm, NULL, NULL, &dmGrads[f]);CHKERRQ(ierr);

        // if there is a dm for this field (does not have to be)
        if (dmGrads[f]) {
            Vec grad;
            ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
            ierr = DMGetGlobalVector(dmGrads[f], &grad);CHKERRQ(ierr);
            // this function looks like it only compute the gradient for the field specified in fvm
            ierr = DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad);CHKERRQ(ierr);
            /* Communicate gradient values */
            ierr = DMGetLocalVector(dmGrads[f], &locGrads[f]);CHKERRQ(ierr);
            ierr = DMGlobalToLocalBegin(dmGrads[f], grad, INSERT_VALUES, locGrads[f]);CHKERRQ(ierr);
            ierr = DMGlobalToLocalEnd(dmGrads[f], grad, INSERT_VALUES, locGrads[f]);CHKERRQ(ierr);
            ierr = DMRestoreGlobalVector(dmGrads[f], &grad);CHKERRQ(ierr);
        }
    }

    // repeat the setup for the aux variables
    ierr = PetscCalloc1(naf, &dmAuxGrads);CHKERRQ(ierr);
    ierr = PetscCalloc1(naf, &locAuxGrads);CHKERRQ(ierr);

    // for each field compute the gradient in the localGrads vector
    for (PetscInt f = 0; f < naf; f++){
        PetscFV fvm;
        ierr = DMGetField(dmAux, f, NULL, (PetscObject*)&fvm);CHKERRQ(ierr);

        // Get the needed auxDm
        //ierr = DMPlexGetGradientDM(dm, fvm, &dmGrad);CHKERRQ(ierr);
        // this call replaces DMPlexGetGradientDM.  DMPlexGetGradientDM only grabs the first created dm, while this creates one (correct size) for each field
        ierr = DMPlexGetDataFVM_MulfiField(dmAux, fvm, NULL, NULL, &dmAuxGrads[f]);CHKERRQ(ierr);

        // if there is a dm grad for this field (does not have to be)
        if (dmAuxGrads[f]) {
            Vec grad;
            ierr = DMPlexGetHeightStratum(dmAux, 1, &fStart, &fEnd);CHKERRQ(ierr);
            ierr = DMGetGlobalVector(dmAuxGrads[f], &grad);CHKERRQ(ierr);
            // this function looks like it only compute the gradient for the field specified in fvm
            ierr = DMPlexReconstructGradientsFVM_MulfiField(dmAux, fvm,  locA, grad);CHKERRQ(ierr);

            /* Communicate gradient values */
            ierr = DMGetLocalVector(dmAuxGrads[f], &locAuxGrads[f]);CHKERRQ(ierr);
            ierr = DMGlobalToLocalBegin(dmAuxGrads[f], grad, INSERT_VALUES, locAuxGrads[f]);CHKERRQ(ierr);
            ierr = DMGlobalToLocalEnd(dmAuxGrads[f], grad, INSERT_VALUES, locAuxGrads[f]);CHKERRQ(ierr);

            // fill the boundary conditions
            /* this is a similar call to DMPlexInsertBoundaryValues, but for gradients */
            ierr = ABLATE_FillGradientBoundary(dmAux, fvm, locA, locAuxGrads[f]);CHKERRQ(ierr);
            ierr = DMRestoreGlobalVector(dmAuxGrads[f], &grad);CHKERRQ(ierr);
        }
    }


    /* Loop over chunks */
    numCells      = cEnd - cStart;
    numChunks     = 1;
    cellChunkSize = numCells/numChunks;
    faceChunkSize = (fEnd - fStart)/numChunks;
    numChunks     = PetscMin(1,numCells);
    for (chunk = 0; chunk < numChunks; ++chunk) {
        PetscScalar     *elemVec, *fluxL, *fluxR;
        PetscReal       *vol;
        PetscFVFaceGeom *fgeom;
        PetscInt         cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;
        PetscInt         fS = fStart+chunk*faceChunkSize, fE = PetscMin(fS+faceChunkSize, fEnd), numFaces = 0, face;

        /* Size up the flux arrays */
        ierr = DMPlexGetFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
        ierr = DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
        ierr = DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);
        ierr = PetscArrayzero(fluxL, numFaces*totDim);CHKERRQ(ierr);
        ierr = PetscArrayzero(fluxR, numFaces*totDim);CHKERRQ(ierr);

        // extract all of the field locations
        ierr = ABLATE_DMPlexGetFaceFields(dm, fS, fE, locX, faceGeometryFVM, cellGeometryFVM, locGrads, &numFaces, &uL, &uR, &gradL, &gradR, PETSC_TRUE);CHKERRQ(ierr);
        ierr = ABLATE_DMPlexGetFaceFields(dmAux, fS, fE, locA, faceGeometryFVM, cellGeometryFVM, locAuxGrads, &numFaces, &auxL, &auxR, &gradAuxL, &gradAuxR, PETSC_FALSE);CHKERRQ(ierr);// NOTE: aux fields are not projected

        /* Loop over each rhs function */
        for (PetscInt d = 0; d < numberFunctionDescriptions; ++d) {
            PetscObject  obj;
            PetscClassId id;
            PetscBool    fimp;
            PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

            PetscInt f = functionDescriptions[d].field;
            ierr = PetscDSGetImplicit(ds, f, &fimp);CHKERRQ(ierr);
            if (isImplicit != fimp) continue;
            ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
            ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);

            PetscFV fv = (PetscFV) obj;

            Ne = numFaces;
            /* Riemann solve over faces (need fields at face centroids) */
            /*   We need to evaluate FE fields at those coordinates */
            ierr = ABLATE_PetscFVIntegrateRHSFunction(&functionDescriptions[d], fv, ds,dsAux, Ne, fgeom, vol, uL, uR, gradL, gradR, auxL, auxR, gradAuxL, gradAuxR, fluxL, fluxR);CHKERRQ(ierr);
        }

        /* Loop over domain and add each face flux back to the cell center*/
        {
            PetscScalar *fa;
            PetscInt     iface;

            ierr = VecGetArray(locF, &fa);CHKERRQ(ierr);
            for (PetscInt f = 0; f < nf; ++f) {
                PetscFV      fv;
                PetscObject  obj;
                PetscClassId id;
                PetscInt     foff, pdim;

                ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
                ierr = PetscDSGetFieldOffset(ds, f, &foff);CHKERRQ(ierr);
                ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
                if (id != PETSCFV_CLASSID) continue;
                fv   = (PetscFV) obj;
                ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
                /* Accumulate fluxes to cells */
                for (face = fS, iface = 0; face < fE; ++face) {
                    const PetscInt *scells;
                    PetscScalar    *fL = NULL, *fR = NULL;
                    PetscInt        ghost, d, nsupp, nchild;

                    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
                    ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
                    ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
                    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
                    ierr = DMPlexGetSupport(dm, face, &scells);CHKERRQ(ierr);
                    ierr = DMLabelGetValue(ghostLabel,scells[0],&ghost);CHKERRQ(ierr);
                    if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, scells[0], f, fa, &fL);CHKERRQ(ierr);}
                    ierr = DMLabelGetValue(ghostLabel,scells[1],&ghost);CHKERRQ(ierr);
                    if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, scells[1], f, fa, &fR);CHKERRQ(ierr);}
                    for (d = 0; d < pdim; ++d) {
                        if (fL) fL[d] -= fluxL[iface*totDim+foff+d];
                        if (fR) fR[d] += fluxR[iface*totDim+foff+d];
                    }
                    ++iface;
                }
            }
            ierr = VecRestoreArray(locF, &fa);CHKERRQ(ierr);
        }

        /* Handle time derivative */
        if (locX_t) {
            PetscScalar *x_t, *fa;

            ierr = VecGetArray(locF, &fa);CHKERRQ(ierr);
            ierr = VecGetArray(locX_t, &x_t);CHKERRQ(ierr);
            for (PetscInt f = 0; f < nf; ++f) {
                PetscFV      fv;
                PetscObject  obj;
                PetscClassId id;
                PetscInt     pdim, d;

                ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
                ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
                if (id != PETSCFV_CLASSID) continue;
                fv   = (PetscFV) obj;
                ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
                for (c = cS; c < cE; ++c) {
                    const PetscInt cell = cells ? cells[c] : c;
                    PetscScalar   *u_t, *r;

                    if (ghostLabel) {
                        PetscInt ghostVal;

                        ierr = DMLabelGetValue(ghostLabel, cell, &ghostVal);CHKERRQ(ierr);
                        if (ghostVal > 0) continue;
                    }
                    ierr = DMPlexPointLocalFieldRead(dm, cell, f, x_t, &u_t);CHKERRQ(ierr);
                    ierr = DMPlexPointLocalFieldRef(dm, cell, f, fa, &r);CHKERRQ(ierr);
                    for (d = 0; d < pdim; ++d) r[d] += u_t[d];
                }
            }
            ierr = VecRestoreArray(locX_t, &x_t);CHKERRQ(ierr);
            ierr = VecRestoreArray(locF, &fa);CHKERRQ(ierr);
        }
        // cleanup
        ierr = ABLATE_DMPlexRestoreFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, NULL, &numFaces, &uL, &uR, &gradL, &gradR);CHKERRQ(ierr);
        ierr = ABLATE_DMPlexRestoreFaceFields(dmAux, fS, fE, locA, NULL, faceGeometryFVM, cellGeometryFVM, NULL, &numFaces, &auxL, &auxR, &gradAuxL, &gradAuxR);CHKERRQ(ierr);
        ierr = DMPlexRestoreFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
        ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
        ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);

        // clean up the field grads
        for (PetscInt f = 0; f < nf; f++){
            // if there is a grad dm for this field (does not have to be), restore
            if (dmGrads[f]) {
                ierr = DMRestoreLocalVector(dmGrads[f], &locGrads[f]);CHKERRQ(ierr);
            }
        }
        // clean up the aux field grads
        for (PetscInt f = 0; f < naf; f++){
            // if there is a grad dm for this field (does not have to be), restore
            if (dmAuxGrads[f]) {
                ierr = DMRestoreLocalVector(dmAuxGrads[f], &locAuxGrads[f]);CHKERRQ(ierr);
            }
        }
    }
    ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);


    PetscFree(dmGrads);
    PetscFree(locGrads);
    PetscFunctionReturn(0);
}


/**
 * Helper function to march over each cell and update the aux Fields
 * @param flow
 * @param time
 * @param locXVec
 * @param updateFunction
 * @return
 */
PetscErrorCode FVFlowUpdateAuxFieldsFV(DM dm, DM auxDM, PetscReal time, Vec locXVec, Vec locAuxField, PetscInt numberUpdateFunctions, FVAuxFieldUpdateFunction* updateFunctions, void** data) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Extract the cell geometry, and the dm that holds the information
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar *cellGeomArray;
    ierr = DMPlexGetGeometryFVM(dm, NULL, &cellGeomVec, NULL);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);

    // Assume that the euler field is always zero
    PetscInt EULER = 0;

    // Get the cell start and end for the fv cells
    PetscInt cellStart, cellEnd;
    ierr = DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd);CHKERRQ(ierr);

    // extract the low flow and aux fields
    const PetscScalar      *locFlowFieldArray;
    ierr = VecGetArrayRead(locXVec, &locFlowFieldArray);CHKERRQ(ierr);

    PetscScalar     *localAuxFlowFieldArray;
    ierr = VecGetArray(locAuxField, &localAuxFlowFieldArray);CHKERRQ(ierr);

    // Get the cell dim
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // March over each cell volume
    for (PetscInt c = cellStart; c < cellEnd; ++c) {
        PetscFVCellGeom       *cellGeom;
        const PetscReal           *fieldValues;
        PetscReal           *auxValues;

        ierr = DMPlexPointLocalRead(dmCell, c, cellGeomArray, &cellGeom);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(dm, c, EULER, locFlowFieldArray, &fieldValues);CHKERRQ(ierr);

        for (PetscInt auxFieldIndex = 0; auxFieldIndex < numberUpdateFunctions; auxFieldIndex ++){
            ierr = DMPlexPointLocalFieldRef(auxDM, c, auxFieldIndex, localAuxFlowFieldArray, &auxValues);CHKERRQ(ierr);

            // If an update function was passed
            if (updateFunctions[auxFieldIndex]){
                updateFunctions[auxFieldIndex](time, dim, cellGeom, fieldValues, auxValues, data[auxFieldIndex]);
            }
        }
    }

    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &locFlowFieldArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(locAuxField, &localAuxFlowFieldArray);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ABLATE_DMPlexComputePointResidual_Internal(FVMRHSPointFunctionDescription *functionDescriptions, PetscInt numberFunctionDescription, DM dm, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    /* FEM+FVM */
    PetscInt         cStart, cEnd;
    const PetscInt  *cells = NULL;
    ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);

    /* 1: Get sizes from dm and dmAux */
    PetscSection     section    = NULL;
    DMLabel          ghostLabel = NULL;
    PetscDS          ds         = NULL;
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
    ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds);CHKERRQ(ierr);

    // determine the number of fields and the totDim
    PetscInt nf, totDim;
    ierr = PetscDSGetNumFields(ds, &nf);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    Vec locA = NULL;
    DM dmAux = NULL;
    PetscDS dsAux = NULL;
    PetscInt totDimAux = 0;
    PetscInt naf = 0;

    ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
    if (locA) {
        PetscInt subcell;
        ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
        ierr = DMGetEnclosurePoint(dmAux, dm, DM_ENC_UNKNOWN, cStart, &subcell);CHKERRQ(ierr);
        ierr = DMGetCellDS(dmAux, subcell, &dsAux);CHKERRQ(ierr);
        ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
        ierr = PetscDSGetNumFields(dsAux, &naf);CHKERRQ(ierr);
    }

    /* 2: Get geometric data */
    // We can use a single call for the geometry data because it does not depend on the fv object
    Vec cellGeometryVec;
    const PetscScalar* cellGeometryArray;
    DM dmCell;
    ierr = DMPlexGetGeometryFVM(dm, NULL, &cellGeometryVec, NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryVec, &cellGeometryArray);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeometryVec, &dmCell);CHKERRQ(ierr);

    /* 3: Get access to the raw u and aux vec */
    const PetscScalar* locXArray;
    const PetscScalar* locAArray;
    ierr = VecGetArrayRead(locX, &locXArray);CHKERRQ(ierr);
    if (locA) {
        ierr = VecGetArrayRead(locA, &locAArray);CHKERRQ(ierr);
    }
    // Get write access to the f array
    PetscScalar * fArray;
    ierr = VecGetArray(locF, &fArray);CHKERRQ(ierr);

    // Create the required offset arrays. These are sized for the max possible value
    PetscInt *uOff, *aOff = NULL;
    PetscCalloc1(nf, &uOff);
    PetscCalloc1(naf, &aOff);
    PetscScalar* fScratch;
    PetscCalloc1(totDim, &fScratch);

    // get the spacial dim and the number of components per field (used for flux)
    PetscInt dim;
    ierr = PetscDSGetSpatialDimension(ds, &dim);CHKERRQ(ierr);

    // March over each cell
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // make sure that this is not a ghost cell
        if (ghostLabel) {
            PetscInt ghostVal;

            ierr = DMLabelGetValue(ghostLabel, cell, &ghostVal);CHKERRQ(ierr);
            if (ghostVal > 0) continue;
        }

        // extract the point locations for this cell
        const PetscFVCellGeom *cg;
        const PetscScalar *u;
        PetscScalar *rhs;
        ierr = DMPlexPointLocalRead(dmCell, cell, cellGeometryArray, &cg);
        ierr = DMPlexPointLocalRead(dm, cell, locXArray, &u);
        ierr = DMPlexPointLocalRef(dm, cell, fArray, &rhs);

        // if there is an aux field, get it
        const PetscScalar *a = NULL;
        if (locA){
            ierr = DMPlexPointLocalRead(dmAux, cell, locAArray, &a);
        }

        // get the full set of offsets from the ds
        PetscInt * uOffTotal;
        ierr = PetscDSGetComponentOffsets(ds, &uOffTotal);CHKERRQ(ierr);
        PetscInt *auxOffTotal = NULL;
        if (dsAux) {
            ierr = PetscDSGetComponentOffsets(dsAux, &auxOffTotal);CHKERRQ(ierr);
        }

        // March over each functionDescriptions
        for (PetscInt f =0; f < numberFunctionDescription; f++){
            // copy over the offsets for each
            for (PetscInt i =0; i < functionDescriptions[f].numberInputFields; i++){
                uOff[i] = uOffTotal[functionDescriptions[f].inputFields[i]];
            }
            if (dsAux) {
                for (PetscInt i =0; i < functionDescriptions[f].numberAuxFields; i++){
                    aOff[i] = uOffTotal[functionDescriptions[f].auxFields[i]];
                }
            }

            // (PetscInt dim, const PetscFVCellGeom *cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void *ctx)
            ierr = functionDescriptions[f].function(dim, cg, uOff, u, aOff, a, fScratch, functionDescriptions[f].context);

            // copy over each result flux field
            PetscInt r = 0;
            for(PetscInt ff = 0; ff < functionDescriptions[f].numberFields; ff++){
                PetscInt fieldSize, fieldOffset;
                ierr = PetscDSGetFieldSize(ds, functionDescriptions[f].fields[ff], &fieldSize);
                ierr = PetscDSGetFieldOffset(ds, functionDescriptions[f].fields[ff], &fieldOffset);
                for (PetscInt d = 0; d < fieldSize; ++d) {
                    rhs[fieldOffset + d] += fScratch[r++];
                }
            }
        }
    }

    // cleanup
    PetscFree(uOff);
    PetscFree(aOff);
    PetscFree(fScratch);

    ierr = VecRestoreArray(locF, &fArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locX, &locXArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locA, &locAArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeometryVec, &cellGeometryArray);CHKERRQ(ierr);
    ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ABLATE_DMPlexComputeRHSJacobianFVM(FVMRHSPointJacobianDescription *functionDescriptions, PetscInt numberFunctionDescription, DM dm, PetscReal t, Vec u, Mat aMat, Mat pMat) {
    PetscFunctionBeginUser;

    IS             cellIS;
    DM             plex;
    PetscInt       depth;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = DMConvert(dm,DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
    ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
    if (!cellIS) {
        ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);
    }
    ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
    ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
    if (!cellIS) {
        ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);
    }

    /* FEM+FVM */
    PetscInt         cStart, cEnd;
    const PetscInt  *cells = NULL;
    ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);

    /* 1: Get sizes from dm and dmAux */
    PetscSection     section    = NULL;
    DMLabel          ghostLabel = NULL;
    PetscDS          ds         = NULL;
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
    ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds);CHKERRQ(ierr);

    // determine the number of fields and the totDim
    PetscInt nf, totDim;
    ierr = PetscDSGetNumFields(ds, &nf);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);

    /* 2: Get geometric data */
    // We can use a single call for the geometry data because it does not depend on the fv object
    Vec cellGeometryVec;
    const PetscScalar* cellGeometryArray;
    DM dmCell;
    ierr = DMPlexGetGeometryFVM(dm, NULL, &cellGeometryVec, NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryVec, &cellGeometryArray);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeometryVec, &dmCell);CHKERRQ(ierr);

    /* 3: Get access to the raw u and aux vec */
    const PetscScalar* globXArray;
    ierr = VecGetArrayRead(u, &globXArray);CHKERRQ(ierr);

    // size up the off
    PetscInt *uOff;
    PetscCalloc1(nf, &uOff);

    // get the full set of offsets from the ds
    PetscInt * uOffTotal;
    ierr = PetscDSGetComponentOffsets(ds, &uOffTotal);CHKERRQ(ierr);

    // get a work array
    PetscScalar jacobianArray
    ierr = DMGetWorkArray(dm, totDim MPIU_SCALAR, uL);CHKERRQ(ierr);

    // March over each cell
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // make sure that this is not a ghost cell
        if (ghostLabel) {
            PetscInt ghostVal;

            ierr = DMLabelGetValue(ghostLabel, cell, &ghostVal);CHKERRQ(ierr);
            if (ghostVal > 0) continue;
        }

        // read the global field
        const PetscScalar *u;
        ierr = DMPlexPointGlobalRead(dm, cell, globXArray, &u);

        // March over each functionDescriptions
        for(PetscInt f =0; f < numberFunctionDescription; f++){
            // copy over the offsets for each
            for (PetscInt i =0; i < functionDescriptions[f].numberFields; i++){
                uOff[i] = uOffTotal[functionDescriptions[f].fields[i]];
            }

            // (PetscInt dim, const PetscFVCellGeom *cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void *ctx)
            ierr = functionDescriptions[f].function(dim, cg, uOff, u, aOff, a, fScratch, functionDescriptions[f].context);


//            DMPlexMatSetClosure(dm, section, globalSection, Jac, cell, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);}

            // copy over each result flux field
            PetscInt r = 0;
            for(PetscInt ff = 0; ff < functionDescriptions[f].numberFields; ff++){
                PetscInt fieldSize, fieldOffset;
                ierr = PetscDSGetFieldSize(ds, functionDescriptions[f].fields[ff], &fieldSize);
                ierr = PetscDSGetFieldOffset(ds, functionDescriptions[f].fields[ff], &fieldOffset);
                for (PetscInt d = 0; d < fieldSize; ++d) {
                    rhs[fieldOffset + d] += fScratch[r++];
                }
            }
        }


    }


    // cleanup
    PetscFree(uOff);
    ierr = VecRestoreArrayRead(u, &globXArray);CHKERRQ(ierr);
    ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


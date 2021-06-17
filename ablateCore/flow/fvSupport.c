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

PetscErrorCode ABLATE_DMPlexTSComputeRHSFunctionFVM(FVMRHSFunctionDescription functionDescription[], PetscInt numberFunctionDescription, DM dm, PetscReal time, Vec locX, Vec F, void *user)
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
    ierr = ABLATE_DMPlexComputeResidual_Internal(functionDescription, numberFunctionDescription, plex, cellIS, time, locX, NULL, time, locF, user);CHKERRQ(ierr);
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

  Level: developer

.seealso: DMPlexGetCellFields()
@*/
static PetscErrorCode ABLATE_DMPlexGetFaceFields(DM dm, PetscInt fStart, PetscInt fEnd, Vec locX, Vec faceGeometry, Vec cellGeometry, const Vec* locGrads, PetscInt *Nface, PetscScalar **uL, PetscScalar **uR)
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
        ierr = PetscCalloc1(Nf, &lgrads);;CHKERRQ(ierr);
        ierr = PetscCalloc1(Nf, &dmGrads);;CHKERRQ(ierr);
        for (f = 0; f < Nf; ++f) {
            if(locGrads[f]) {
                ierr = VecGetArrayRead(locGrads[f], &lgrads[f]);CHKERRQ(ierr);
                ierr = VecGetDM(locGrads[f], &dmGrads[f]);CHKERRQ(ierr);
            }
        }
    }

    /* Right now just eat the extra work for FE (could make a cell loop) */
    for (face = fStart, iface = 0; face < fEnd; ++face) {
        const PetscInt        *cells;
        PetscFVFaceGeom       *fg;
        PetscFVCellGeom       *cgL, *cgR;
        PetscScalar           *xL, *xR, *gL, *gR;
        PetscScalar           *uLl = *uL, *uRl = *uR;
        PetscInt               ghost, nsupp, nchild;

        ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
        ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
        ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);

        // march over each field
        for (f = 0; f < Nf; ++f) {
            PetscInt off;

            ierr = PetscDSGetComponentOffset(prob, f, &off);CHKERRQ(ierr);

            PetscFV  fv;
            PetscInt numComp, c;

            ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fv);CHKERRQ(ierr);
            ierr = PetscFVGetNumComponents(fv, &numComp);CHKERRQ(ierr);
            ierr = DMPlexPointLocalFieldRead(dm, cells[0], f, x, &xL);CHKERRQ(ierr);
            ierr = DMPlexPointLocalFieldRead(dm, cells[1], f, x, &xR);CHKERRQ(ierr);
            if(dmGrads[f]){
                PetscReal dxL[3], dxR[3];

                ierr = DMPlexPointLocalRead(dmGrads[f], cells[0], lgrads[f], &gL);CHKERRQ(ierr);
                ierr = DMPlexPointLocalRead(dmGrads[f], cells[1], lgrads[f], &gR);CHKERRQ(ierr);
                DMPlex_WaxpyD_Internal(dim, -1, cgL->centroid, fg->centroid, dxL);
                DMPlex_WaxpyD_Internal(dim, -1, cgR->centroid, fg->centroid, dxR);
                for (c = 0; c < numComp; ++c) {
                    uLl[iface*Nc+off+c] = xL[c] + DMPlex_DotD_Internal(dim, &gL[c*dim], dxL);
                    uRl[iface*Nc+off+c] = xR[c] + DMPlex_DotD_Internal(dim, &gR[c*dim], dxR);
                }
            } else {
                for (c = 0; c < numComp; ++c) {
                    uLl[iface*Nc+off+c] = xL[c];
                    uRl[iface*Nc+off+c] = xR[c];
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
                if(locGrads[f]) {
                    ierr = VecRestoreArrayRead(locGrads[f], &lgrads[f]);CHKERRQ(ierr);
                }
            }
        ierr = PetscFree(lgrads);;CHKERRQ(ierr);
    }
    ierr = PetscFree(isFE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


/*
 * Private function to compute the rhs based upon a FVMRHSFunctionDescription.
 *
 * Not it is assumed that the fvm object's field is the same one as in functionDescription
  neighborVol[f*2+0] contains the left  geom
  neighborVol[f*2+1] contains the right geom
*/
static PetscErrorCode ABLATE_PetscFVIntegrateRHSFunction(FVMRHSFunctionDescription functionDescription[], PetscFV fvm, PetscDS prob, PetscInt numberFaces, PetscFVFaceGeom *fgeom, PetscReal *neighborVol,
                                                         PetscScalar uL[], PetscScalar uR[], PetscScalar fluxL[], PetscScalar fluxR[])
{
    void              *rctx;
    PetscScalar       *flux = fvm->fluxWork;
    PetscErrorCode     ierr;

    PetscFunctionBegin;

    // Get the total number of components (in all fields)
    PetscInt nCompTot;
    ierr = PetscDSGetTotalComponents(prob, &nCompTot);CHKERRQ(ierr);
    PetscInt totalDim;//This is usually the same?
    ierr = PetscDSGetTotalDimension(prob, &totalDim);CHKERRQ(ierr);

    // create the required offset arrays
    PetscInt *uOff;
    PetscCalloc1(functionDescription->numberInputFields, &uOff);

    // Get the full set of offsets from the ds
    PetscInt * uOffTotal;
    ierr = PetscDSGetComponentOffsets(prob, &uOffTotal);CHKERRQ(ierr);
    for(PetscInt f =0; f < functionDescription->numberInputFields; f++){
        uOff[f] = uOffTotal[functionDescription->inputFields[f]];
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
//        ( PetscInt dim, const PetscFVFaceGeom* fg, const PetscFVCellGeom* cgL, const PetscFVCellGeom* cgR,
//        const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar* gradL[], const PetscScalar* gradR[],
//        const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar* gradAuxL[], const PetscScalar* gradAuxR[],
//        PetscScalar* flux, void* ctx);
        ierr = functionDescription->function(dim, &fgeom[f], NULL, NULL,
                              uOff, &uL[f*nCompTot], &uR[f*nCompTot], NULL, NULL,
                              NULL, NULL, NULL, NULL, NULL,
                              flux, functionDescription->context);CHKERRQ(ierr);
//        (*riemann)(dim, pdim, fgeom[f].centroid, fgeom[f].normal, &uL[f*nCompTot], &uR[f*Nc], numConstants, constants, flux, rctx);
        for (PetscInt d = 0; d < fluxDim; ++d) {
            fluxL[f*totalDim+fluxOffset+d] = flux[d] / neighborVol[f*2+0];
            fluxR[f*totalDim+fluxOffset+d] = flux[d] / neighborVol[f*2+1];
        }
    }

    PetscFree(uOff);
    PetscFunctionReturn(0);
}

PetscErrorCode ABLATE_DMPlexComputeResidual_Internal(FVMRHSFunctionDescription functionDescription[], PetscInt numberFunctionDescription, DM dm, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
    DM_Plex         *mesh       = (DM_Plex *) dm->data;
    const char      *name       = "Residual";
    DM               dmAux      = NULL;
    DM               *dmGrads     = NULL;
    DMLabel          ghostLabel = NULL;
    PetscDS          ds         = NULL;
    PetscDS          dsAux      = NULL;
    PetscSection     section    = NULL;
    PetscBool        isImplicit = (locX_t || time == PETSC_MIN_REAL) ? PETSC_TRUE : PETSC_FALSE;
    PetscFVCellGeom *cgeomFVM   = NULL;
    PetscFVFaceGeom *fgeomFVM   = NULL;
    DMField          coordField = NULL;
    Vec *locGrads =NULL;  // each field will have a separate local gradient vector
    Vec              locA, cellGeometryFVM = NULL, faceGeometryFVM = NULL;
    PetscScalar     *u = NULL, *u_t, *a, *uL, *uR;
    IS               chunkIS;
    const PetscInt  *cells;
    PetscInt         cStart, cEnd, numCells;
    PetscInt         Nf, f, totDim, totDimAux, numChunks, cellChunkSize, faceChunkSize, chunk, fStart, fEnd;
    PetscInt         maxDegree = PETSC_MAX_INT;
    PetscQuadrature  affineQuad = NULL, *quads = NULL;
    PetscFEGeom     *affineGeom = NULL, **geoms = NULL;
    PetscErrorCode   ierr;

    PetscFunctionBegin;
    /* FEM+FVM */
    ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    /* 1: Get sizes from dm and dmAux */
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
    ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
    if (locA) {
        PetscInt subcell;
        ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
        ierr = DMGetEnclosurePoint(dmAux, dm, DM_ENC_UNKNOWN, cStart, &subcell);CHKERRQ(ierr);
        ierr = DMGetCellDS(dmAux, subcell, &dsAux);CHKERRQ(ierr);
        ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    }
    /* 2: Get geometric data */
    // We can use a single call for the geometry data because it does not depend on the fv object
    ierr = DMPlexGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);
    /* Reconstruct and limit cell gradients */

    // Get the dm grad for each field
    ierr = PetscCalloc1(Nf, &dmGrads);CHKERRQ(ierr);
    ierr = PetscCalloc1(Nf, &locGrads);CHKERRQ(ierr);

    // for each field compute the gradient in the localGrads vector
    for(f = 0; f < Nf; f++){
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

    /* Handle non-essential (e.g. outflow) boundary values */
    // TODO: this will need to be updated to add each gradient field for each vector
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, time, faceGeometryFVM, cellGeometryFVM, NULL);CHKERRQ(ierr);

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
        ierr = ABLATE_DMPlexGetFaceFields(dm, fS, fE, locX, faceGeometryFVM, cellGeometryFVM, locGrads, &numFaces, &uL, &uR);CHKERRQ(ierr);


        /* Loop over each rhs function */
        for (f = 0; f < numberFunctionDescription; ++f) {
            PetscObject  obj;
            PetscClassId id;
            PetscBool    fimp;
            PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

            ierr = PetscDSGetImplicit(ds, f, &fimp);CHKERRQ(ierr);
            if (isImplicit != fimp) continue;
            ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
            ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);

            PetscFV fv = (PetscFV) obj;

            Ne = numFaces;
            /* Riemann solve over faces (need fields at face centroids) */
            /*   We need to evaluate FE fields at those coordinates */
            ierr = ABLATE_PetscFVIntegrateRHSFunction(&functionDescription[f], fv, ds, Ne, fgeom, vol, uL, uR, fluxL, fluxR);CHKERRQ(ierr);
        }
        /* Loop over domain and add each face flux back to the cell center*/
        {
            PetscScalar *fa;
            PetscInt     iface;

            ierr = VecGetArray(locF, &fa);CHKERRQ(ierr);
            for (f = 0; f < Nf; ++f) {
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
            for (f = 0; f < Nf; ++f) {
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
        ierr = DMPlexRestoreFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, NULL, &numFaces, &uL, &uR);CHKERRQ(ierr);
        ierr = DMPlexRestoreFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
        ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
        ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);

        for(f = 0; f < Nf; f++){
            // if there is a grad dm for this field (does not have to be), restore
            if (dmGrads[f]) {
                ierr = DMRestoreLocalVector(dmGrads[f], &locGrads[f]);CHKERRQ(ierr);
            }
        }


    }
    ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);


    PetscFree(dmGrads);
    PetscFree(locGrads);
    PetscFunctionReturn(0);
}
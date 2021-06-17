#include "fvSupport.h"
#include <inttypes.h>
#include <petsc/private/dmpleximpl.h>

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

PetscErrorCode ABLATE_DMPlexTSComputeRHSFunctionFVM(DM dm, PetscReal time, Vec locX, Vec F, void *user)
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
    ierr = ABLATE_DMPlexComputeResidual_Internal(plex, cellIS, time, locX, NULL, time, locF, user);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(plex, locF, ADD_VALUES, F);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(plex, locF, ADD_VALUES, F);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(plex, &locF);CHKERRQ(ierr);
    ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode ABLATE_DMPlexComputeResidual_Internal(DM dm, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
    DM_Plex         *mesh       = (DM_Plex *) dm->data;
    const char      *name       = "Residual";
    DM               dmAux      = NULL;
    DM               dmGrad     = NULL;
    DMLabel          ghostLabel = NULL;
    PetscDS          ds         = NULL;
    PetscDS          dsAux      = NULL;
    PetscSection     section    = NULL;
    PetscBool        useFEM     = PETSC_FALSE;
    PetscBool        useFVM     = PETSC_FALSE;
    PetscBool        isImplicit = (locX_t || time == PETSC_MIN_REAL) ? PETSC_TRUE : PETSC_FALSE;
    PetscFV          fvm        = NULL;
    PetscFVCellGeom *cgeomFVM   = NULL;
    PetscFVFaceGeom *fgeomFVM   = NULL;
    DMField          coordField = NULL;
    Vec              locA, cellGeometryFVM = NULL, faceGeometryFVM = NULL, grad, locGrad = NULL;
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
    /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
    /* TODO The FVM geometry is over-manipulated. Make the precalc functions return exactly what we need */
    /* FEM+FVM */
    ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    /* 1: Get sizes from dm and dmAux */
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
    ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
    if (locA) {
        PetscInt subcell;
        ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
        ierr = DMGetEnclosurePoint(dmAux, dm, DM_ENC_UNKNOWN, cStart, &subcell);CHKERRQ(ierr);
        ierr = DMGetCellDS(dmAux, subcell, &dsAux);CHKERRQ(ierr);
        ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    }
    /* 2: Get geometric data */
    for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscBool    fimp;

        ierr = PetscDSGetImplicit(ds, f, &fimp);CHKERRQ(ierr);
        if (isImplicit != fimp) continue;
        ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID) {useFEM = PETSC_TRUE;}
        if (id == PETSCFV_CLASSID) {useFVM = PETSC_TRUE; fvm = (PetscFV) obj;}
    }
    if (useFEM) {
        ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
        ierr = DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
        if (maxDegree <= 1) {
            ierr = DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad);CHKERRQ(ierr);
            if (affineQuad) {
                ierr = DMSNESGetFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
            }
        } else {
            ierr = PetscCalloc2(Nf,&quads,Nf,&geoms);CHKERRQ(ierr);
            for (f = 0; f < Nf; ++f) {
                PetscObject  obj;
                PetscClassId id;
                PetscBool    fimp;

                ierr = PetscDSGetImplicit(ds, f, &fimp);CHKERRQ(ierr);
                if (isImplicit != fimp) continue;
                ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
                ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
                if (id == PETSCFE_CLASSID) {
                    PetscFE fe = (PetscFE) obj;

                    ierr = PetscFEGetQuadrature(fe, &quads[f]);CHKERRQ(ierr);
                    ierr = PetscObjectReference((PetscObject)quads[f]);CHKERRQ(ierr);
                    ierr = DMSNESGetFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);
                }
            }
        }
    }
    if (useFVM) {
        ierr = DMPlexGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL);CHKERRQ(ierr);
        ierr = VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
        ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);
        /* Reconstruct and limit cell gradients */
        ierr = DMPlexGetGradientDM(dm, fvm, &dmGrad);CHKERRQ(ierr);
        if (dmGrad) {
            ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
            ierr = DMGetGlobalVector(dmGrad, &grad);CHKERRQ(ierr);
            ierr = DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad);CHKERRQ(ierr);
            /* Communicate gradient values */
            ierr = DMGetLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);
            ierr = DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
            ierr = DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
            ierr = DMRestoreGlobalVector(dmGrad, &grad);CHKERRQ(ierr);
        }
        /* Handle non-essential (e.g. outflow) boundary values */
        ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, time, faceGeometryFVM, cellGeometryFVM, locGrad);CHKERRQ(ierr);
    }
    /* Loop over chunks */
    if (useFEM) {ierr = ISCreate(PETSC_COMM_SELF, &chunkIS);CHKERRQ(ierr);}
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

        /* Extract field coefficients */
        if (useFEM) {
            ierr = ISGetPointSubrange(chunkIS, cS, cE, cells);CHKERRQ(ierr);
            ierr = DMPlexGetCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
            ierr = DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
            ierr = PetscArrayzero(elemVec, numCells*totDim);CHKERRQ(ierr);
        }
        if (useFVM) {
            ierr = DMPlexGetFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR);CHKERRQ(ierr);
            ierr = DMPlexGetFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
            ierr = DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
            ierr = DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);
            ierr = PetscArrayzero(fluxL, numFaces*totDim);CHKERRQ(ierr);
            ierr = PetscArrayzero(fluxR, numFaces*totDim);CHKERRQ(ierr);
        }
        /* TODO We will interlace both our field coefficients (u, u_t, uL, uR, etc.) and our output (elemVec, fL, fR). I think this works */
        /* Loop over fields */
        for (f = 0; f < Nf; ++f) {
            PetscObject  obj;
            PetscClassId id;
            PetscBool    fimp;
            PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

            key.field = f;
            ierr = PetscDSGetImplicit(ds, f, &fimp);CHKERRQ(ierr);
            if (isImplicit != fimp) continue;
            ierr = PetscDSGetDiscretization(ds, f, &obj);CHKERRQ(ierr);
            ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
            if (id == PETSCFE_CLASSID) {
                PetscFE         fe = (PetscFE) obj;
                PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
                PetscFEGeom    *chunkGeom = NULL;
                PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
                PetscInt        Nq, Nb;

                ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
                ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
                ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
                blockSize = Nb;
                batchSize = numBlocks * blockSize;
                ierr      = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
                numChunks = numCells / (numBatches*batchSize);
                Ne        = numChunks*numBatches*batchSize;
                Nr        = numCells % (numBatches*batchSize);
                offset    = numCells - Nr;
                /* Integrate FE residual to get elemVec (need fields at quadrature points) */
                /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
                ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
                ierr = PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, u_t, dsAux, a, t, elemVec);CHKERRQ(ierr);
                ierr = PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
                ierr = PetscFEIntegrateResidual(ds, key, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, dsAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
                ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
            } else if (id == PETSCFV_CLASSID) {
                PetscFV fv = (PetscFV) obj;

                Ne = numFaces;
                /* Riemann solve over faces (need fields at face centroids) */
                /*   We need to evaluate FE fields at those coordinates */
                ierr = PetscFVIntegrateRHSFunction(fv, ds, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR);CHKERRQ(ierr);
            } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
        }
        /* Loop over domain */
        if (useFEM) {
            /* Add elemVec to locX */
            for (c = cS; c < cE; ++c) {
                const PetscInt cell = cells ? cells[c] : c;
                const PetscInt cind = c - cStart;

                if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]);CHKERRQ(ierr);}
                if (ghostLabel) {
                    PetscInt ghostVal;

                    ierr = DMLabelGetValue(ghostLabel,cell,&ghostVal);CHKERRQ(ierr);
                    if (ghostVal > 0) continue;
                }
                ierr = DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
            }
        }
        if (useFVM) {
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
        if (useFEM) {
            ierr = DMPlexRestoreCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
            ierr = DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
        }
        if (useFVM) {
            ierr = DMPlexRestoreFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR);CHKERRQ(ierr);
            ierr = DMPlexRestoreFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
            ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
            ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);
            if (dmGrad) {ierr = DMRestoreLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);}
        }
    }
    if (useFEM) {ierr = ISDestroy(&chunkIS);CHKERRQ(ierr);}
    ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);

    if (useFEM) {
        ierr = DMPlexComputeBdResidual_Internal(dm, locX, locX_t, t, locF, user);CHKERRQ(ierr);

        if (maxDegree <= 1) {
            ierr = DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
            ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
        } else {
            for (f = 0; f < Nf; ++f) {
                ierr = DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);
                ierr = PetscQuadratureDestroy(&quads[f]);CHKERRQ(ierr);
            }
            ierr = PetscFree2(quads,geoms);CHKERRQ(ierr);
        }
    }

    /* FEM */
    /* 1: Get sizes from dm and dmAux */
    /* 2: Get geometric data */
    /* 3: Handle boundary values */
    /* 4: Loop over domain */
    /*   Extract coefficients */
    /* Loop over fields */
    /*   Set tiling for FE*/
    /*   Integrate FE residual to get elemVec */
    /*     Loop over subdomain */
    /*       Loop over quad points */
    /*         Transform coords to real space */
    /*         Evaluate field and aux fields at point */
    /*         Evaluate residual at point */
    /*         Transform residual to real space */
    /*       Add residual to elemVec */
    /* Loop over domain */
    /*   Add elemVec to locX */

    /* FVM */
    /* Get geometric data */
    /* If using gradients */
    /*   Compute gradient data */
    /*   Loop over domain faces */
    /*     Count computational faces */
    /*     Reconstruct cell gradient */
    /*   Loop over domain cells */
    /*     Limit cell gradients */
    /* Handle boundary values */
    /* Loop over domain faces */
    /*   Read out field, centroid, normal, volume for each side of face */
    /* Riemann solve over faces */
    /* Loop over domain faces */
    /*   Accumulate fluxes to cells */
    /* TODO Change printFEM to printDisc here */
    if (mesh->printFEM) {
        Vec         locFbc;
        PetscInt    pStart, pEnd, p, maxDof;
        PetscScalar *zeroes;

        ierr = VecDuplicate(locF,&locFbc);CHKERRQ(ierr);
        ierr = VecCopy(locF,locFbc);CHKERRQ(ierr);
        ierr = PetscSectionGetChart(section,&pStart,&pEnd);CHKERRQ(ierr);
        ierr = PetscSectionGetMaxDof(section,&maxDof);CHKERRQ(ierr);
        ierr = PetscCalloc1(maxDof,&zeroes);CHKERRQ(ierr);
        for (p = pStart; p < pEnd; p++) {
            ierr = VecSetValuesSection(locFbc,section,p,zeroes,INSERT_BC_VALUES);CHKERRQ(ierr);
        }
        ierr = PetscFree(zeroes);CHKERRQ(ierr);
        ierr = DMPrintLocalVec(dm, name, mesh->printTol, locFbc);CHKERRQ(ierr);
        ierr = VecDestroy(&locFbc);CHKERRQ(ierr);
    }
    ierr = PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
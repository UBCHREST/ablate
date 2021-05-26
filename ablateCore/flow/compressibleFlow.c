#include "compressibleFlow.h"
#include "fvSupport.h"

static const char *compressibleFlowComponentNames[TOTAL_COMPRESSIBLE_FLOW_COMPONENTS + 1] = {"rho", "rhoE", "rhoU", "rhoV", "rhoW", "unknown"};
static const char *compressibleAuxComponentNames[TOTAL_COMPRESSIBLE_AUX_COMPONENTS + 1] = {"T", "vel", "unknown"};
const char *compressibleFlowParametersTypeNames[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS + 1] = {"cfl", "gamma", "Rgas", "k", "mu", "unknown"};

static inline void NormVector(PetscInt dim, const PetscReal* in, PetscReal* out){
    PetscReal mag = 0.0;
    for (PetscInt d=0; d< dim; d++) {
        mag += in[d]*in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d=0; d< dim; d++) {
        out[d] = in[d]/mag;
    }
}

static inline PetscReal MagVector(PetscInt dim, const PetscReal* in){
    PetscReal mag = 0.0;
    for (PetscInt d=0; d< dim; d++) {
        mag += in[d]*in[d];
    }
    return PetscSqrtReal(mag);
}

/**
 * Helper function to march over each cell and update the aux Fields
 * @param flow
 * @param time
 * @param locXVec
 * @param updateFunction
 * @return
 */
static PetscErrorCode FVFlowUpdateAuxFieldsFV(FlowData flowData, PetscReal time, Vec locXVec, FVAuxFieldUpdateFunction* updateFunctions) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    // Extract the cell geometry, and the dm that holds the information
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar *cellGeomArray;
    ierr = DMPlexGetGeometryFVM(flowData->dm, NULL, &cellGeomVec, NULL);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);

    // Assume that the euler field is always zero
    PetscInt EULER = 0;

    // Get the cell start and end for the fv cells
    PetscInt cellStart, cellEnd;
    ierr = DMPlexGetHeightStratum(flowData->dm, EULER, &cellStart, &cellEnd);CHKERRQ(ierr);

    // extract the low flow and aux fields
    const PetscScalar      *locFlowFieldArray;
    ierr = VecGetArrayRead(locXVec, &locFlowFieldArray);CHKERRQ(ierr);

    PetscScalar     *localAuxFlowFieldArray;
    ierr = VecGetArray(flowData->auxField, &localAuxFlowFieldArray);CHKERRQ(ierr);

    // Get the cell dim
    PetscInt dim;
    ierr = DMGetDimension(flowData->dm, &dim);CHKERRQ(ierr);

    // March over each cell volume
    for (PetscInt c = cellStart; c < cellEnd; ++c) {
        PetscFVCellGeom       *cellGeom;
        const PetscReal           *fieldValues;
        PetscReal           *auxValues;

        ierr = DMPlexPointLocalRead(dmCell, c, cellGeomArray, &cellGeom);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(flowData->dm, c, EULER, locFlowFieldArray, &fieldValues);CHKERRQ(ierr);

        for (PetscInt auxFieldIndex = 0; auxFieldIndex < flowData->numberAuxFields; auxFieldIndex ++){
            ierr = DMPlexPointLocalFieldRef(flowData->auxDm, c, auxFieldIndex, localAuxFlowFieldArray, &auxValues);CHKERRQ(ierr);

            // If an update function was passed
            if (updateFunctions[auxFieldIndex]){
                updateFunctions[auxFieldIndex](flowData, time, dim, cellGeom, fieldValues, auxValues);
            }
        }
    }

    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &locFlowFieldArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(flowData->auxField, &localAuxFlowFieldArray);CHKERRQ(ierr);

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
static PetscErrorCode FVFlowFillGradientBoundary(DM dm, PetscFV auxFvm, Vec localXVec, Vec gradLocalVec){
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

/**
 * Function to get the density, velocity, and energy from the conserved variables
 * @return
 */
static void DecodeState(PetscInt dim, const PetscReal* conservedValues,  const PetscReal *normal, PetscReal gamma, PetscReal* density,
                                  PetscReal* normalVelocity, PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* M, PetscReal* p){
    // decode
    *density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE]/(*density);

    // Get the velocity in this direction
    (*normalVelocity) = 0.0;
    for (PetscInt d =0; d < dim; d++){
        velocity[d] = conservedValues[RHOU + d]/(*density);
        (*normalVelocity) += velocity[d]*normal[d];
    }

    // get the speed
    PetscReal speed = MagVector(dim, velocity);

    // assumed eos
    (*internalEnergy) = (totalEnergy) - 0.5 * speed * speed;
    *p = (gamma - 1.0)*(*density)*(*internalEnergy);
    *a = PetscSqrtReal(gamma*(*p)/(*density));

    *M = (*normalVelocity)/(*a);
}

PetscErrorCode CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVelL, const PetscReal * gradVelR, PetscReal* tau){
    PetscFunctionBeginUser;
    // pre compute the div of the velocity field
    PetscReal divVel = 0.0;
    for (PetscInt c =0; c < dim; ++c){
        divVel += 0.5*(gradVelL[c*dim + c] + gradVelR[c*dim + c]);
    }

    // March over each velocity component, u, v, w
    for (PetscInt c =0; c < dim; ++c){
        // March over each physical coordinate coordinate
        for (PetscInt d =0; d < dim; ++d) {
            if (d == c) {
                // for the xx, yy, zz, components
                tau[c*dim + d] = 2.0 * mu * (0.5 * (gradVelL[c * dim + d] + gradVelR[c * dim + d]) - divVel / 3.0);
            } else {
                // for xy, xz, etc
                tau[c*dim + d]  = mu *( 0.5 * (gradVelL[c * dim + d] + gradVelR[c * dim + d]) + 0.5 * (gradVelL[d * dim + c] + gradVelR[d * dim + c]));
            }
        }
    }
    PetscFunctionReturn(0);
}

void CompressibleFlowComputeEulerFlux(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, area, norm);
    const PetscReal areaMag = MagVector(dim, area);

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal ML;
    PetscReal pL;
    DecodeState(dim, xL, norm, flowParameters->gamma, &densityL, &normalVelocityL, velocityL, &internalEnergyL, &aL, &ML, &pL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal MR;
    PetscReal pR;
    DecodeState(dim, xR, norm, flowParameters->gamma, &densityR, &normalVelocityR, velocityR, &internalEnergyR, &aR, &MR, &pR);

    PetscReal sPm;
    PetscReal sPp;
    PetscReal sMm;
    PetscReal sMp;

    flowParameters->fluxDifferencer(MR, &sPm, &sMm, ML, &sPp, &sMp);

    flux[RHO] = (sMm* densityR * aR + sMp* densityL * aL) * areaMag;

    PetscReal velMagR = MagVector(dim, velocityR);
    PetscReal HR = internalEnergyR + velMagR*velMagR/2.0 + pR/densityR;
    PetscReal velMagL = MagVector(dim, velocityL);
    PetscReal HL = internalEnergyL + velMagL*velMagL/2.0 + pL/densityL;

    flux[RHOE] = (sMm * densityR * aR * HR + sMp * densityL * aL * HL) * areaMag;

    for (PetscInt n =0; n < dim; n++) {
        flux[RHOU + n] = (sMm * densityR * aR * velocityR[n] + sMp * densityL * aL * velocityL[n]) * areaMag + (pR*sPm + pL*sPp) * area[n];
    }
}

/*
 * Compute the rhs source terms for diffusion processes
 */
static PetscErrorCode CompressibleFlowDiffusionSourceRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, FlowData flowData) {
    PetscFunctionBeginUser;
    // Call the flux calculation
    PetscErrorCode ierr;

    // get the fvm fields, for now we assume we need grad for all
    PetscFV auxFvm[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
    DM auxFieldGradDM[TOTAL_COMPRESSIBLE_AUX_COMPONENTS]; /* dm holding the grad information */
    for (PetscInt af =0; af < TOTAL_COMPRESSIBLE_AUX_COMPONENTS; af++){
        ierr = DMGetField(flowData->auxDm, af, NULL, (PetscObject*)&auxFvm[af]);CHKERRQ(ierr);

        // Get the needed auxDm
        ierr = DMPlexGetDataFVM_MulfiField(flowData->auxDm, auxFvm[af], NULL, NULL, &auxFieldGradDM[af]);CHKERRQ(ierr);
        if (!auxFieldGradDM[af]){
            SETERRQ(PetscObjectComm((PetscObject)flowData->auxDm), PETSC_ERR_ARG_WRONGSTATE, "The FVM method for aux variables must support computing gradients.");
        }
    }

    // get the dim
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // Get the locXArray
    const PetscScalar *locAuxArray;
    ierr = VecGetArrayRead(flowData->auxField, &locAuxArray);CHKERRQ(ierr);

    // Get the fvm face and cell geometry
    Vec cellGeomVec = NULL;/* vector of structs related to cell geometry*/
    Vec faceGeomVec = NULL;/* vector of structs related to face geometry*/

    // extract the fvm data
    ierr = DMPlexGetGeometryFVM(flowData->dm, &faceGeomVec, &cellGeomVec, NULL);CHKERRQ(ierr);

    // get the dm for each geom type
    DM dmFaceGeom, dmCellGeom;
    ierr = VecGetDM(faceGeomVec, &dmFaceGeom);CHKERRQ(ierr);
    ierr = VecGetDM(cellGeomVec, &dmCellGeom);CHKERRQ(ierr);

    // extract the arrays for the face and cell geom, along with their dm
    const PetscScalar *faceGeomArray, *cellGeomArray;
    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);

    // Obtaining local cell and face ownership
    PetscInt faceStart, faceEnd;
    PetscInt cellStart, cellEnd;
    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd);CHKERRQ(ierr);

    // get the ghost label
    DMLabel ghostLabel;
    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);

    // extract the localFArray from the locFVec
    PetscScalar *fa;
    Vec locFVec;
    ierr = DMGetLocalVector(dm, &locFVec);CHKERRQ(ierr);
    ierr = VecZeroEntries(locFVec);CHKERRQ(ierr);
    ierr = VecGetArray(locFVec, &fa);CHKERRQ(ierr);

    // create a global and local grad vector for the auxField
    Vec gradAuxGlobalVec[TOTAL_COMPRESSIBLE_AUX_COMPONENTS], gradAuxLocalVec[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
    const PetscScalar *localGradArray[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
    for (PetscInt af =0; af < TOTAL_COMPRESSIBLE_AUX_COMPONENTS; af++) {
        ierr = DMCreateGlobalVector(auxFieldGradDM[af], &gradAuxGlobalVec[af]);CHKERRQ(ierr);

        // compute the global grad values
        ierr = DMPlexReconstructGradientsFVM_MulfiField(flowData->auxDm, auxFvm[af], flowData->auxField, gradAuxGlobalVec[af]);CHKERRQ(ierr);

        // Map to a local grad vector
        ierr = DMCreateLocalVector(auxFieldGradDM[af], &gradAuxLocalVec[af]);CHKERRQ(ierr);

        PetscInt size;
        VecGetSize(gradAuxGlobalVec[af], &size);

        ierr = DMGlobalToLocalBegin(auxFieldGradDM[af], gradAuxGlobalVec[af], INSERT_VALUES, gradAuxLocalVec[af]);CHKERRQ(ierr);
        ierr = DMGlobalToLocalEnd(auxFieldGradDM[af], gradAuxGlobalVec[af], INSERT_VALUES, gradAuxLocalVec[af]);CHKERRQ(ierr);

        // fill the boundary conditions
        ierr = FVFlowFillGradientBoundary(flowData->auxDm, auxFvm[af], flowData->auxField, gradAuxLocalVec[af]);CHKERRQ(ierr);

        // access the local vector
        ierr = VecGetArrayRead(gradAuxLocalVec[af], &localGradArray[af]);CHKERRQ(ierr);
    }

    // Get the flow parameters
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;

    // march over each face
    for (PetscInt face = faceStart; face < faceEnd; ++face) {
        PetscFVFaceGeom       *fg;
        PetscFVCellGeom       *cgL, *cgR;
        const PetscScalar           *gradL[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
        const PetscScalar           *gradR[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];

        // make sure that this is a valid face to check
        PetscInt  ghost, nsupp, nchild;
        ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
        ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
        if (ghost >= 0 || nsupp > 2 || nchild > 0){
            continue;// skip this face
        }

        // get the face geometry
        ierr = DMPlexPointLocalRead(dmFaceGeom, face, faceGeomArray, &fg);CHKERRQ(ierr);

        // Get the left and right cells for this face
        const PetscInt        *faceCells;
        ierr = DMPlexGetSupport(dm, face, &faceCells);CHKERRQ(ierr);

        // get the cell geom for the left and right faces
        ierr = DMPlexPointLocalRead(dmCellGeom, faceCells[0], cellGeomArray, &cgL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalRead(dmCellGeom, faceCells[1], cellGeomArray, &cgR);CHKERRQ(ierr);

        // extract the cell grad
        for (PetscInt af =0; af < TOTAL_COMPRESSIBLE_AUX_COMPONENTS; af++) {
            ierr = DMPlexPointLocalRead(auxFieldGradDM[af], faceCells[0], localGradArray[af], &gradL[af]);CHKERRQ(ierr);
            ierr = DMPlexPointLocalRead(auxFieldGradDM[af], faceCells[1], localGradArray[af], &gradR[af]);CHKERRQ(ierr);
        }
        PetscInt euler = 0;

        // extract the field values
        PetscScalar *auxL, *auxR;
        ierr = DMPlexPointLocalFieldRead(flowData->auxDm, faceCells[0], euler, locAuxArray, &auxL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(flowData->auxDm, faceCells[1], euler, locAuxArray, &auxR);CHKERRQ(ierr);

        // Add to the source terms of f
        PetscScalar    *fL = NULL, *fR = NULL;
        ierr = DMLabelGetValue(ghostLabel,faceCells[0],&ghost);CHKERRQ(ierr);
        if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, faceCells[0], euler, fa, &fL);CHKERRQ(ierr);}
        ierr = DMLabelGetValue(ghostLabel,faceCells[1],&ghost);CHKERRQ(ierr);
        if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, faceCells[1], euler, fa, &fR);CHKERRQ(ierr);}

        {// everything here should be a separate function to allow reuse of the code to march over faces
            // Compute the stress tensor tau
            PetscReal tau[9];// Maximum size without symmetry
            ierr = CompressibleFlowComputeStressTensor(dim, flowParameters->mu, gradL[VEL], gradR[VEL], tau);CHKERRQ(ierr);

            // for each velocity component
            for (PetscInt c =0; c < dim; ++c) {
                PetscReal viscousFlux = 0.0;

                // March over each direction
                for (PetscInt d = 0; d < dim; ++d) {
                    viscousFlux += -fg->normal[d]*tau[c*dim + d];// This is tau[c][d]
                }

                // add in the contribution
                if (fL){
                    fL[RHOU + c] -= viscousFlux/cgL->volume;
                }
                if (fR){
                    fR[RHOU + c] += viscousFlux/cgR->volume;
                }
            }

            // energy equation
            for (PetscInt d = 0; d < dim; ++d) {
                PetscReal heatFlux = 0.0;
                // add in the contributions for this viscous terms
                for (PetscInt c =0; c < dim; ++c){
                    heatFlux += 0.5*(auxL[VEL + c] + auxR[VEL+c]) * tau[d * dim + c];
                }

               // heat conduction (-k dT/dx - k dT/dy - k dT/dz) . n A
                heatFlux += +flowParameters->k * 0.5 * (gradL[T][d] + gradR[T][d]);

                // Multiply by the area normal
                heatFlux *= -fg->normal[d];

                if (fL) {
                    fL[RHOE] -= heatFlux / cgL->volume;
                }
                if (fR) {
                    fR[RHOE] += heatFlux / cgR->volume;
                }
            }
        }
    }

    // Add the new locFVec to the globFVec
    ierr = VecRestoreArray(locFVec, &fa);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm, locFVec, ADD_VALUES, globFVec);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm, locFVec, ADD_VALUES, globFVec);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm, &locFVec);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(flowData->auxField, &locAuxArray);CHKERRQ(ierr);

    for (PetscInt af =0; af < TOTAL_COMPRESSIBLE_AUX_COMPONENTS; af++) {
        // restore the arrays
        ierr = VecRestoreArrayRead(gradAuxLocalVec[af], &localGradArray[af]);CHKERRQ(ierr);

        // destroy grad vectors
        ierr = VecDestroy(&gradAuxGlobalVec[af]);CHKERRQ(ierr);
        ierr = VecDestroy(&gradAuxLocalVec[af]);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode CompressibleFlowRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    FlowData flowData = (FlowData)ctx;
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;

    // compute the euler flux across each face (note CompressibleFlowComputeEulerFlux has already been registered)
    ierr = DMPlexTSComputeRHSFunctionFVM(dm, time, locXVec, globFVec, ctx);CHKERRQ(ierr);

    // if there are any coefficients for diffusion, compute diffusion
    if (flowParameters->k || flowParameters->mu){
        // update any aux fields
        ierr = FVFlowUpdateAuxFieldsFV(flowData, time, locXVec, flowParameters->auxFieldUpdateFunctions);CHKERRQ(ierr);

        // compute the RHS sources
        ierr = CompressibleFlowDiffusionSourceRHSFunctionLocal(dm, time, locXVec, globFVec, ctx);CHKERRQ(ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode CompressibleFlow_SetupDiscretization(FlowData flowData, DM* dm) {
    PetscInt dim;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
    const PetscInt ghostCellDepth = 1;
    {// Make sure that the flow is setup distributed
        DM dmDist;
        ierr = DMSetBasicAdjacency(*dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
        ierr = DMPlexDistribute(*dm, ghostCellDepth, NULL, &dmDist);CHKERRQ(ierr);
        if (dmDist) {
            ierr = DMDestroy(dm);CHKERRQ(ierr);
            *dm   = dmDist;
        }
    }

    // create any ghost cells that are needed
    {
        DM gdm;
        ierr = DMPlexConstructGhostCells(*dm, NULL, NULL, &gdm);CHKERRQ(ierr);
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm   = gdm;
    }

    //Store the field data
    flowData->dm = *dm;
    ierr = DMSetApplicationContext(flowData->dm, flowData);CHKERRQ(ierr);

    // Determine the number of dimensions
    ierr = DMGetDimension(flowData->dm, &dim);CHKERRQ(ierr);

    // Register a single field
    PetscInt numberComponents = 2+dim;
    ierr = FlowRegisterField(flowData, "euler", "euler", numberComponents, FV);CHKERRQ(ierr);

    // Name each of the components, this is used by some of the output fields
    PetscFV fvm;
    ierr = DMGetField(flowData->dm,0, NULL, (PetscObject*)&fvm);CHKERRQ(ierr);
    for (PetscInt c =0; c <numberComponents; c++){
        ierr = PetscFVSetComponentName(fvm, c, compressibleFlowComponentNames[c]);CHKERRQ(ierr);
    }

    // Create the discrete systems for the DM based upon the fields added to the DM
    ierr = FlowFinalizeRegisterFields(flowData);CHKERRQ(ierr);

    // Register the aux fields, note the order should match the enum
    ierr = FlowRegisterAuxField(flowData, compressibleAuxComponentNames[T], compressibleAuxComponentNames[T], 1, FV);CHKERRQ(ierr);
    ierr = FlowRegisterAuxField(flowData, compressibleAuxComponentNames[VEL], compressibleAuxComponentNames[VEL], dim, FV);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode UpdateAuxTemperatureField(FlowData flowData, PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField){
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE]/density;

    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;

    // Get the velocity in this direction
    PetscReal speedSquare = 0.0;
    for (PetscInt d =0; d < dim; d++){
        speedSquare += PetscSqr(conservedValues[RHOU + d]/density);
    }

    // assumed eos
    PetscReal internalEnergy = (totalEnergy) - 0.5 * speedSquare;
    PetscReal p = (flowParameters->gamma - 1.0)*density*internalEnergy;

    auxField[T] = p/(flowParameters->Rgas*density);
    PetscFunctionReturn(0);
}

static PetscErrorCode UpdateAuxVelocityField(FlowData flowData, PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField){
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[RHO];

    for (PetscInt d =0; d < dim; d++){
        auxField[d] = conservedValues[RHOU + d] / density;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode CompressibleFlow_StartProblemSetup(FlowData flowData, PetscInt num, PetscScalar values[]) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);CHKERRQ(ierr);

    // Set the flux calculator solver for each component
    ierr = PetscDSSetRiemannSolver(prob, 0, CompressibleFlowComputeEulerFlux);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, flowData);CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);

    // Create the euler data
    EulerFlowData *data;
    PetscNew(&data);
    flowData->data =data;

    // make sure we have enough params
    if (num <= RGAS){
        SETERRQ(PetscObjectComm((PetscObject) flowData->dm), PETSC_ERR_ARG_OUTOFRANGE, "insufficient number of arguments for compressible flow");
    }

    data->cfl = values[CFL];
    data->gamma = values[GAMMA];
    data->Rgas = values[RGAS];
    data->k = num > K ? values[K] : 0.0;
    data->mu = num > MU ? values[MU] : 0.0;

    // Set the update fields
    data->auxFieldUpdateFunctions[T] = UpdateAuxTemperatureField;
    data->auxFieldUpdateFunctions[VEL] = UpdateAuxVelocityField;

    const char *prefix;
    ierr = DMGetOptionsPrefix(flowData->dm, &prefix);CHKERRQ(ierr);

    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)flowData->dm),prefix,"Compressible Flow Options",NULL);CHKERRQ(ierr);

    // setup the flux differencer
    PetscFunctionList fluxDifferencerList;
    ierr = FluxDifferencerListGet(&fluxDifferencerList);CHKERRQ(ierr);
    char fluxDiffValue[128] = "ausm";
    ierr = PetscOptionsFList("-flux_diff","Flux differencer","",fluxDifferencerList,fluxDiffValue,fluxDiffValue,sizeof fluxDiffValue,NULL);CHKERRQ(ierr);
    ierr = FluxDifferencerGet(fluxDiffValue, &(data->fluxDifferencer));CHKERRQ(ierr);

    // allow the user to disable the autm
    data->automaticTimeStepCalculator = PETSC_TRUE;
    ierr = PetscOptionsBool("-automaticTimeStepCalculator", "determines if a time step is calculated", NULL, data->automaticTimeStepCalculator, &data->automaticTimeStepCalculator, NULL);CHKERRQ(ierr);

    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode ComputeTimeStep(TS ts, void* context){
    PetscFunctionBeginUser;
    // Get the dm and current solution vector
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    Vec v;
    TSGetSolution(ts, &v);

    // Get the flow param
    FlowData flowData;
    ierr = DMGetApplicationContext(dm, &flowData);CHKERRQ(ierr);
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;

    // Get the fv geom
    PetscReal minCellRadius;
    ierr = DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius);CHKERRQ(ierr);
    PetscInt cStart, cEnd;
    ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    const PetscScalar      *x;
    ierr = VecGetArrayRead(v, &x);CHKERRQ(ierr);

    //Get the dim from the dm
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 *minCellRadius;

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscReal           *xc;
        ierr = DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc);CHKERRQ(ierr);

        if (xc) {  // must be real cell and not ghost
            PetscReal rho = xc[RHO];

            // Compute the kinetic energy
            PetscReal velMag = 0.0;
            for (PetscInt i =0; i < dim; i++){
                velMag += PetscSqr(xc[RHOU + i] / rho);
            }

            PetscReal u = xc[RHOU] / rho;
            PetscReal e = (xc[RHOE] / rho) - 0.5 * velMag;
            PetscReal p = (flowParameters->gamma - 1) * rho * e;


            PetscReal a = PetscSqrtReal(flowParameters->gamma * p / rho);
            PetscReal dt = flowParameters->cfl * dx / (a + PetscAbsReal(u));
            dtMin = PetscMin(dtMin, dt);
        }
    }
    PetscInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);

    PetscReal dtMinGlobal;
    ierr = MPI_Allreduce(&dtMin, &dtMinGlobal, 1,MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)ts));

    ierr = TSSetTimeStep(ts, dtMinGlobal);CHKERRQ(ierr);

    if (PetscIsNanReal(dtMinGlobal)){
        SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_FP, "Invalid timestep selected for flow");
    }

    ierr = VecRestoreArrayRead(v, &x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode CompressibleFlow_CompleteProblemSetup(FlowData flowData, TS ts) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    ierr = FlowCompleteProblemSetup(flowData, ts);CHKERRQ(ierr);
    EulerFlowData * compressibleFlowData = (EulerFlowData *)flowData->data;
    if (compressibleFlowData->automaticTimeStepCalculator){
        ierr = FlowRegisterPreStep(flowData, ComputeTimeStep, flowData);CHKERRQ(ierr);
    }

    // Override the DMTSSetRHSFunctionLocal in DMPlexTSComputeRHSFunctionFVM with a function that includes euler and diffusion source terms
    ierr = DMTSSetRHSFunctionLocal(flowData->dm, CompressibleFlowRHSFunctionLocal, flowData);CHKERRQ(ierr);

    // copy over any boundary information from the dm, to the aux dm and set the sideset
    if (flowData->auxDm) {
        PetscDS flowProblem;
        ierr = DMGetDS(flowData->dm, &flowProblem);CHKERRQ(ierr);
        PetscDS auxProblem;
        ierr = DMGetDS(flowData->auxDm, &auxProblem);CHKERRQ(ierr);

        // Get the number of boundary conditions and other info
        PetscInt numberBC;
        ierr = PetscDSGetNumBoundary(flowProblem, &numberBC);CHKERRQ(ierr);
        PetscInt numberAuxFields;
        ierr = PetscDSGetNumFields(auxProblem, &numberAuxFields);CHKERRQ(ierr);

        for (PetscInt bc =0; bc < numberBC; bc++){
            DMBoundaryConditionType type;
            const char* name;
            const char* labelName;
            PetscInt field;
            PetscInt numberIds;
            const PetscInt* ids;

            // Get the boundary
            ierr = PetscDSGetBoundary(flowProblem, bc, &type, &name, &labelName, &field, NULL, NULL, NULL, NULL, &numberIds, &ids, NULL);CHKERRQ(ierr);

            // If this is for euler and DM_BC_NATURAL_RIEMANN add it to the aux
            if (type == DM_BC_NATURAL_RIEMANN && field == 0){
                for (PetscInt af =0; af < numberAuxFields; af++) {
                    ierr = PetscDSAddBoundary(auxProblem, type, name, labelName, af, 0, NULL, NULL, NULL, numberIds, ids, NULL);CHKERRQ(ierr);
                }
            }
        }
    }

    PetscFunctionReturn(0);
}

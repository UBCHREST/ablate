#include "compressibleFlow.h"
#include "fvSupport.h"

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
PetscErrorCode FVFlowUpdateAuxFieldsFV(DM dm, DM auxDM, PetscReal time, Vec locXVec, Vec locAuxField, PetscInt numberUpdateFunctions, FVAuxFieldUpdateFunction* updateFunctions, FlowData_CompressibleFlow data) {
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
                updateFunctions[auxFieldIndex](data, time, dim, cellGeom, fieldValues, auxValues);
            }
        }
    }

    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(locXVec, &locFlowFieldArray);CHKERRQ(ierr);
    ierr = VecRestoreArray(locAuxField, &localAuxFlowFieldArray);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


/**
 * Function to get the density, velocity, and energy from the conserved variables
 * @return
 */
static void DecodeEulerState(FlowData_CompressibleFlow flowData, PetscInt dim, const PetscReal* conservedValues,  const PetscReal *normal, PetscReal* density,
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

    // decode the state in the eos
    flowData->decodeStateFunction(NULL, dim, *density, totalEnergy, velocity, internalEnergy, a, p, flowData->decodeStateFunctionContext);
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
//
///*
// * Compute the rhs source terms for diffusion processes
// */
//PetscErrorCode CompressibleFlowDiffusionSourceRHSFunctionLocal(DM dm, DM auxDM, PetscReal time, Vec locXVec, Vec locAuxVec, Vec globFVec, FlowData_CompressibleFlow flowParameters, FVDiffusionFunction* functions) {
//    PetscFunctionBeginUser;
//    PetscErrorCode ierr;
//
//    // get the fvm fields, for now we assume we need grad for all
//    PetscFV auxFvm[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
//    DM auxFieldGradDM[TOTAL_COMPRESSIBLE_AUX_COMPONENTS]; /* dm holding the grad information */
//    for (PetscInt af =0; af < TOTAL_COMPRESSIBLE_AUX_COMPONENTS; af++){
//        ierr = DMGetField(auxDM, af, NULL, (PetscObject*)&auxFvm[af]);CHKERRQ(ierr);
//
//        // Get the needed auxDm
//        ierr = DMPlexGetDataFVM_MulfiField(auxDM, auxFvm[af], NULL, NULL, &auxFieldGradDM[af]);CHKERRQ(ierr);
//        if (!auxFieldGradDM[af]){
//            SETERRQ(PetscObjectComm((PetscObject)auxDM), PETSC_ERR_ARG_WRONGSTATE, "The FVM method for aux variables must support computing gradients.");
//        }
//    }
//
//    // get the dim
//    PetscInt dim;
//    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
//
//    // Get the locXArray and locAuxArray
//    const PetscScalar *locXArray;
//    ierr = VecGetArrayRead(locXVec, &locXArray);CHKERRQ(ierr);
//    const PetscScalar *locAuxArray;
//    ierr = VecGetArrayRead(locAuxVec, &locAuxArray);CHKERRQ(ierr);
//
//    // Get the fvm face and cell geometry
//    Vec cellGeomVec = NULL;/* vector of structs related to cell geometry*/
//    Vec faceGeomVec = NULL;/* vector of structs related to face geometry*/
//
//    // extract the fvm data
//    ierr = DMPlexGetGeometryFVM(dm, &faceGeomVec, &cellGeomVec, NULL);CHKERRQ(ierr);
//
//    // get the dm for each geom type
//    DM dmFaceGeom, dmCellGeom;
//    ierr = VecGetDM(faceGeomVec, &dmFaceGeom);CHKERRQ(ierr);
//    ierr = VecGetDM(cellGeomVec, &dmCellGeom);CHKERRQ(ierr);
//
//    // extract the arrays for the face and cell geom, along with their dm
//    const PetscScalar *faceGeomArray, *cellGeomArray;
//    ierr = VecGetArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
//    ierr = VecGetArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);
//
//    // Obtaining local cell and face ownership
//    PetscInt faceStart, faceEnd;
//    PetscInt cellStart, cellEnd;
//    ierr = DMPlexGetHeightStratum(dm, 1, &faceStart, &faceEnd);CHKERRQ(ierr);
//    ierr = DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd);CHKERRQ(ierr);
//
//    // get the ghost label
//    DMLabel ghostLabel;
//    ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
//
//    // extract the localFArray from the locFVec
//    PetscScalar *fa;
//    Vec locFVec;
//    ierr = DMGetLocalVector(dm, &locFVec);CHKERRQ(ierr);
//    ierr = VecZeroEntries(locFVec);CHKERRQ(ierr);
//    ierr = VecGetArray(locFVec, &fa);CHKERRQ(ierr);
//
//    // create a global and local grad vector for the auxField
//    Vec gradAuxGlobalVec[TOTAL_COMPRESSIBLE_AUX_COMPONENTS], gradAuxLocalVec[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
//    const PetscScalar *localGradArray[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
//    for (PetscInt af =0; af < TOTAL_COMPRESSIBLE_AUX_COMPONENTS; af++) {
//        ierr = DMCreateGlobalVector(auxFieldGradDM[af], &gradAuxGlobalVec[af]);CHKERRQ(ierr);
//
//        // compute the global grad values
//        ierr = DMPlexReconstructGradientsFVM_MulfiField(auxDM, auxFvm[af], locAuxVec, gradAuxGlobalVec[af]);CHKERRQ(ierr);
//
//        // Map to a local grad vector
//        ierr = DMCreateLocalVector(auxFieldGradDM[af], &gradAuxLocalVec[af]);CHKERRQ(ierr);
//
//        PetscInt size;
//        VecGetSize(gradAuxGlobalVec[af], &size);
//
//        ierr = DMGlobalToLocalBegin(auxFieldGradDM[af], gradAuxGlobalVec[af], INSERT_VALUES, gradAuxLocalVec[af]);CHKERRQ(ierr);
//        ierr = DMGlobalToLocalEnd(auxFieldGradDM[af], gradAuxGlobalVec[af], INSERT_VALUES, gradAuxLocalVec[af]);CHKERRQ(ierr);
//
//        // fill the boundary conditions
//        ierr = FVFlowFillGradientBoundary(auxDM, auxFvm[af], locAuxVec, gradAuxLocalVec[af]);CHKERRQ(ierr);
//
//        // access the local vector
//        ierr = VecGetArrayRead(gradAuxLocalVec[af], &localGradArray[af]);CHKERRQ(ierr);
//    }
//
//    // get the number of fields
//    PetscInt nf;
//    ierr = DMGetNumFields(dm, &nf);CHKERRQ(ierr);
//
//    // march over each face
//    for (PetscInt face = faceStart; face < faceEnd; ++face) {
//        PetscFVFaceGeom       *fg;
//        PetscFVCellGeom       *cgL, *cgR;
//        const PetscScalar           *gradL[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
//        const PetscScalar           *gradR[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];
//
//        // make sure that this is a valid face to check
//        PetscInt  ghost, nsupp, nchild;
//        ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
//        ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
//        ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
//        if (ghost >= 0 || nsupp > 2 || nchild > 0){
//            continue;// skip this face
//        }
//
//        // get the face geometry
//        ierr = DMPlexPointLocalRead(dmFaceGeom, face, faceGeomArray, &fg);CHKERRQ(ierr);
//
//        // Get the left and right cells for this face
//        const PetscInt        *faceCells;
//        ierr = DMPlexGetSupport(dm, face, &faceCells);CHKERRQ(ierr);
//
//        // get the cell geom for the left and right faces
//        ierr = DMPlexPointLocalRead(dmCellGeom, faceCells[0], cellGeomArray, &cgL);CHKERRQ(ierr);
//        ierr = DMPlexPointLocalRead(dmCellGeom, faceCells[1], cellGeomArray, &cgR);CHKERRQ(ierr);
//
//        // extract the cell grad
//        for (PetscInt af =0; af < TOTAL_COMPRESSIBLE_AUX_COMPONENTS; af++) {
//            ierr = DMPlexPointLocalRead(auxFieldGradDM[af], faceCells[0], localGradArray[af], &gradL[af]);CHKERRQ(ierr);
//            ierr = DMPlexPointLocalRead(auxFieldGradDM[af], faceCells[1], localGradArray[af], &gradR[af]);CHKERRQ(ierr);
//        }
//
//        // extract the aux field values// NOTE: DMPlexPointLocalRead is used so we get the entire auxArray
//        PetscScalar *auxL, *auxR;
//        ierr = DMPlexPointLocalRead(auxDM, faceCells[0], locAuxArray, &auxL);CHKERRQ(ierr);
//        ierr = DMPlexPointLocalRead(auxDM, faceCells[1], locAuxArray, &auxR);CHKERRQ(ierr);
//
//        // for each field
//        for(PetscInt f =0; f < nf; f++) {
//            // extract the field values
//            PetscScalar *fieldL, *fieldR;
//            ierr = DMPlexPointLocalFieldRead(dm, faceCells[0], f, locXArray, &fieldL);CHKERRQ(ierr);
//            ierr = DMPlexPointLocalFieldRead(dm, faceCells[1], f, locXArray, &fieldR);CHKERRQ(ierr);
//
//            // Add to the source terms of f
//            PetscScalar *fL = NULL, *fR = NULL;
//            ierr = DMLabelGetValue(ghostLabel, faceCells[0], &ghost);CHKERRQ(ierr);
//            if (ghost <= 0) {
//                ierr = DMPlexPointLocalFieldRef(dm, faceCells[0], f, fa, &fL);CHKERRQ(ierr);
//            }
//            ierr = DMLabelGetValue(ghostLabel, faceCells[1], &ghost);CHKERRQ(ierr);
//            if (ghost <= 0) {
//                ierr = DMPlexPointLocalFieldRef(dm, faceCells[1], f, fa, &fR);CHKERRQ(ierr);
//            }
//
//            if(functions[f]){
//                /*(FlowData_CompressibleFlow flowData, PetscReal time, PetscInt dim, const PetscFVFaceGeom* fg, const PetscFVCellGeom* cgL, const PetscFVCellGeom* cgR,
//                const PetscScalar* fieldL, const PetscScalar* fieldR,const PetscScalar* auxL, const PetscScalar* auxR,
//                const PetscScalar* gradAuxL, const PetscScalar* gradAuxR, PetscScalar* fluxL, PetscScalar* fluxR);*/
//                ierr = functions[f](flowParameters, time, dim, fg, cgL, cgR, fieldL, fieldR, auxL, auxR, gradL, gradR, fL, fR );CHKERRQ(ierr);
//            }
//        }
//    }
//
//    // Add the new locFVec to the globFVec
//    ierr = VecRestoreArray(locFVec, &fa);CHKERRQ(ierr);
//    ierr = DMLocalToGlobalBegin(dm, locFVec, ADD_VALUES, globFVec);CHKERRQ(ierr);
//    ierr = DMLocalToGlobalEnd(dm, locFVec, ADD_VALUES, globFVec);CHKERRQ(ierr);
//    ierr = DMRestoreLocalVector(dm, &locFVec);CHKERRQ(ierr);
//    ierr = VecRestoreArrayRead(cellGeomVec, &cellGeomArray);CHKERRQ(ierr);
//    ierr = VecRestoreArrayRead(faceGeomVec, &faceGeomArray);CHKERRQ(ierr);
//    ierr = VecRestoreArrayRead(locAuxVec, &locAuxArray);CHKERRQ(ierr);
//    ierr = VecRestoreArrayRead(locXVec, &locXArray);CHKERRQ(ierr);
//
//    for (PetscInt af =0; af < TOTAL_COMPRESSIBLE_AUX_COMPONENTS; af++) {
//        // restore the arrays
//        ierr = VecRestoreArrayRead(gradAuxLocalVec[af], &localGradArray[af]);CHKERRQ(ierr);
//
//        // destroy grad vectors
//        ierr = VecDestroy(&gradAuxGlobalVec[af]);CHKERRQ(ierr);
//        ierr = VecDestroy(&gradAuxLocalVec[af]);CHKERRQ(ierr);
//    }
//
//    PetscFunctionReturn(0);
//}


PetscErrorCode CompressibleFlowComputeEulerFlux ( PetscInt dim, const PetscFVFaceGeom* fg, const PetscFVCellGeom* cgL, const PetscFVCellGeom* cgR,
                                                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar uL[], const PetscScalar uR[], const PetscScalar gradL[], const PetscScalar gradR[],
                                                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                                  PetscScalar* flux, void* ctx){
    FlowData_CompressibleFlow flowParameters = (FlowData_CompressibleFlow)ctx;
    PetscFunctionBeginUser;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal ML;
    PetscReal pL;
    DecodeEulerState(flowParameters, dim, uL, norm, &densityL, &normalVelocityL, velocityL, &internalEnergyL, &aL, &ML, &pL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal MR;
    PetscReal pR;
    DecodeEulerState(flowParameters, dim, uR, norm, &densityR, &normalVelocityR, velocityR, &internalEnergyR, &aR, &MR, &pR);

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
        flux[RHOU + n] = (sMm * densityR * aR * velocityR[n] + sMp * densityL * aL * velocityL[n]) * areaMag + (pR*sPm + pL*sPp) * fg->normal[n];
    }

    PetscFunctionReturn(0);
}


PetscErrorCode CompressibleFlowEulerDiffusion(PetscInt dim, const PetscFVFaceGeom* fg, const PetscFVCellGeom* cgL, const PetscFVCellGeom* cgR,
                                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar* fieldL, const PetscScalar* fieldR, const PetscScalar gradL[], const PetscScalar gradR[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar* auxL, const PetscScalar* auxR, const PetscScalar gradAuxL[], const PetscScalar gradAuxR[],
                                              PetscScalar* flux, void* ctx){
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    FlowData_CompressibleFlow flowParameters = (FlowData_CompressibleFlow)ctx;

    // Compute the stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    ierr = CompressibleFlowComputeStressTensor(dim, flowParameters->mu, gradAuxL + aOff_x[VEL], gradAuxR + aOff_x[VEL], tau);
    CHKERRQ(ierr);

    // for each velocity component
    for (PetscInt c = 0; c < dim; ++c) {
        PetscReal viscousFlux = 0.0;

        // March over each direction
        for (PetscInt d = 0; d < dim; ++d) {
            viscousFlux += -fg->normal[d] * tau[c * dim + d];  // This is tau[c][d]
        }

        // add in the contribution
        flux[RHOU + c] = viscousFlux;
    }

    // energy equation
    flux[RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal heatFlux = 0.0;
        // add in the contributions for this viscous terms
        for (PetscInt c = 0; c < dim; ++c) {
            heatFlux += 0.5 * (auxL[aOff[VEL] + c] + auxR[aOff[VEL] + c]) * tau[d * dim + c];
        }

        // heat conduction (-k dT/dx - k dT/dy - k dT/dz) . n A
        heatFlux += +flowParameters->k * 0.5 * (gradAuxL[aOff_x[T] + d] + gradAuxR[aOff_x[T] + d]);

        // Multiply by the area normal
        heatFlux *= -fg->normal[d];

        flux[RHOE] += heatFlux;
    }
    PetscFunctionReturn(0);
}
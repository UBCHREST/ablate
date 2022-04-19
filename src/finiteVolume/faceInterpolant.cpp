#include "faceInterpolant.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::FaceInterpolant::FaceInterpolant(std::shared_ptr<ablate::domain::SubDomain> subDomain, std::shared_ptr<domain::Region> region, Vec faceGeomVec, Vec cellGeomVec)
    : subDomain(subDomain), region(region) {
    auto ds = subDomain->GetDiscreteSystem();
    PetscDSGetTotalDimension(ds, &solTotalSize) >> checkError;
    CreateFaceDm(solTotalSize, subDomain->GetDM(), faceSolutionDm);
    CreateFaceDm(solTotalSize * subDomain->GetDimensions(), subDomain->GetDM(), faceSolutionGradDm);

    auto auxDs = subDomain->GetAuxDiscreteSystem();
    if (auxDs) {
        PetscDSGetTotalDimension(auxDs, &auxTotalSize) >> checkError;
        CreateFaceDm(auxTotalSize, subDomain->GetDM(), faceAuxDm);
        CreateFaceDm(auxTotalSize * subDomain->GetDimensions(), subDomain->GetDM(), faceAuxGradDm);
    }

    // Size up the stencil
    auto dim = subDomain->GetDimensions();
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(subDomain->GetDM(), 1, &fStart, &fEnd) >> checkError;
    stencils.resize(fEnd - fStart);

    // Get the label if the region is provided
    auto dm = subDomain->GetDM();
    DMLabel regionLabel;
    PetscInt regionValue;
    ablate::domain::Region::GetLabel(region, dm, regionLabel, regionValue);

    // Set up the gradient calculator
    PetscFV gradientCalculator;
    PetscFVCreate(PETSC_COMM_SELF, &gradientCalculator) >> checkError;
    // Set least squares as the default type
    PetscFVSetType(gradientCalculator, PETSCFVLEASTSQUARES) >> checkError;
    // Set any other required options
    PetscFVSetFromOptions(gradientCalculator) >> checkError;
    PetscFVSetNumComponents(gradientCalculator, 1) >> checkError;
    PetscFVSetSpatialDimension(gradientCalculator, subDomain->GetDimensions()) >> checkError;

    // Keep track of the current maxFaces
    PetscInt maxFaces = 0;
    std::vector<PetscScalar> dx;

    // Get the geometry for the mesh
    DM faceDM, cellDM;
    VecGetDM(faceGeomVec, &faceDM) >> checkError;
    VecGetDM(cellGeomVec, &cellDM) >> checkError;
    const PetscScalar* cellGeomArray;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    // Compute the stencil for each face
    PetscInt iFace = 0;
    for (PetscInt face = fStart; face < fEnd; face++) {
        auto& stencil = stencils[iFace];
        stencil.faceId = face;

        PetscFVFaceGeom* fg;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;
        PetscArraycpy(stencil.area, fg->normal, dim);
        PetscArraycpy(stencil.normal, fg->normal, dim);
        utilities::MathUtilities::NormVector(dim, stencil.normal);
        PetscArraycpy(stencil.centroid, fg->centroid, dim);

        // Get all nodes in this face
        PetscInt numberNodes;
        PetscInt* faceNodes = nullptr;
        DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &numberNodes, &faceNodes) >> checkError;

        // For each node get the cells that connect to it
        for (PetscInt n = 0; n < numberNodes; n++) {
            PetscInt numberCells;
            PetscInt* nodeCells = nullptr;
            DMPlexGetTransitiveClosure(dm, faceNodes[n * 2], PETSC_FALSE, &numberCells, &nodeCells) >> checkError;

            for (PetscInt c = 0; c < numberCells; c++) {
                PetscInt cell = nodeCells[c * 2];
                // Make sure that cell is in this region and is a cell
                PetscInt cellHeight;
                DMPlexGetPointHeight(dm, cell, &cellHeight) >> checkError;
                if (cellHeight != 0) {
                    continue;
                }
                //                if (regionLabel) {  // TODO: remove this region check, do a ds check instead
                //                    PetscInt labelValue;
                //                    DMLabelGetValue(regionLabel, cell, &labelValue) >> checkError;
                //                    if (labelValue != regionValue) {
                //                        continue;
                //                    }
                //                }

                stencil.stencil.push_back(cell);
            }

            // cleanup
            DMPlexRestoreTransitiveClosure(dm, faceNodes[n * 2], PETSC_FALSE, &numberCells, &nodeCells) >> checkError;
        }

        // cleanup
        DMPlexRestoreTransitiveClosure(dm, face, PETSC_TRUE, &numberNodes, &faceNodes) >> checkError;

        // Clean up the stencil to remove duplicates
        std::sort(stencil.stencil.begin(), stencil.stencil.end());
        stencil.stencil.erase(std::unique(stencil.stencil.begin(), stencil.stencil.end()), stencils[iFace].stencil.end());

        // ignore cell if there are no stencils
        stencil.stencilSize = (PetscInt)stencil.stencil.size();
        if (stencil.stencilSize) {
            // for now, set the interpolant weights to be the average of the two faces
            stencil.weights.resize(stencil.stencilSize, 0.0);
            stencil.gradientWeights.resize(stencil.stencilSize * subDomain->GetDimensions(), 0.0);
            if (stencil.stencilSize * dim > (PetscInt)dx.size()) {
                dx.resize(stencil.stencilSize * dim);
            }

            // Get the support for this face
            PetscInt numberNeighborCells;
            const PetscInt* neighborCells;
            DMPlexGetSupportSize(dm, face, &numberNeighborCells) >> ablate::checkError;
            DMPlexGetSupport(dm, face, &neighborCells) >> ablate::checkError;
            // Set the stencilWeight
            PetscReal sum = 0.0;
            for (PetscInt c = 0; c < numberNeighborCells; c++) {
                for (PetscInt i = 0; i < stencil.stencilSize; i++) {
                    if (neighborCells[c] == stencil.stencil[i]) {
                        stencil.weights[i] = 1.0;
                        sum += 1.0;
                    }
                }
            }
            ablate::utilities::MathUtilities::ScaleVector(stencil.stencilSize, stencil.weights.data(), 1.0 / sum);

            // Compute gradients
            if (stencil.stencilSize > maxFaces) {
                maxFaces = stencil.stencilSize;
                PetscFVLeastSquaresSetMaxFaces(gradientCalculator, maxFaces) >> checkError;
            }

            // compute the distance between the cell centers and the face
            for (PetscInt n = 0; n < stencil.stencilSize; n++) {
                PetscFVCellGeom* cg;
                DMPlexPointLocalRead(cellDM, stencil.stencil[n], cellGeomArray, &cg);
                for (PetscInt d = 0; d < dim; ++d) {
                    dx[n * dim + d] = cg->centroid[d] - fg->centroid[d];
                }
            }

            PetscFVComputeGradient(gradientCalculator, stencil.stencilSize, &dx[0], stencil.gradientWeights.data()) >> checkError;
        }
        iFace++;
    }
    // clean up the geom
    PetscFVDestroy(&gradientCalculator);
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
}

ablate::finiteVolume::FaceInterpolant::~FaceInterpolant() {
    if (faceSolutionDm) {
        DMDestroy(&faceSolutionDm);
    }
    if (faceSolutionGradDm) {
        DMDestroy(&faceSolutionGradDm);
    }
    if (faceAuxDm) {
        DMDestroy(&faceAuxDm);
    }
    if (faceAuxGradDm) {
        DMDestroy(&faceAuxGradDm);
    }
}

void ablate::finiteVolume::FaceInterpolant::CreateFaceDm(PetscInt totalDim, DM dm, DM& newDm) {
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> checkError;

    // create a face solution dm for that is the required number of variables per face
    PetscSection solutionSection;
    DMClone(dm, &newDm) >> checkError;
    PetscSectionCreate(PetscObjectComm((PetscObject)dm), &solutionSection) >> checkError;

    PetscSectionSetChart(solutionSection, fStart, fEnd) >> checkError;
    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscSectionSetDof(solutionSection, f, totalDim) >> checkError;
    }

    PetscSectionSetUp(solutionSection) >> checkError;
    DMSetLocalSection(newDm, solutionSection) >> checkError;
    PetscSectionDestroy(&solutionSection) >> checkError;
}

void ablate::finiteVolume::FaceInterpolant::GetInterpolatedFaceVectors(Vec solutionVec, Vec auxVec, Vec& faceSolutionVec, Vec& faceAuxVec, Vec& faceSolutionGradVec, Vec& faceAuxGradVec) {
    // Compute the stencil for each face
    auto dim = subDomain->GetDimensions();
    PetscInt fEnd;
    DMPlexGetHeightStratum(subDomain->GetDM(), 1, &globalFaceStart, &fEnd) >> checkError;
    stencils.resize(fEnd - globalFaceStart);

    // Size the return vectors
    DMGetLocalVector(faceSolutionDm, &faceSolutionVec) >> checkError;
    DMGetLocalVector(faceSolutionGradDm, &faceAuxVec) >> checkError;
    if (auxTotalSize) {
        DMGetLocalVector(faceAuxDm, &faceSolutionGradVec) >> checkError;
        DMGetLocalVector(faceAuxGradDm, &faceAuxGradVec) >> checkError;
    }

    // Extract each of the dms needed
    DM solutionDm, auxDm;
    VecGetDM(solutionVec, &solutionDm) >> checkError;
    if (auxVec) {
        VecGetDM(auxVec, &auxDm) >> checkError;
    }

    // Get the arrays
    const PetscScalar* solutionArray;
    VecGetArrayRead(solutionVec, &solutionArray);
    const PetscScalar* auxArray;
    if (auxTotalSize) {
        VecGetArrayRead(auxVec, &auxArray);
    }

    PetscScalar* faceSolutionArray;
    PetscScalar* faceAuxArray;
    VecGetArray(faceSolutionVec, &faceSolutionArray);
    if (auxTotalSize) {
        VecGetArray(faceAuxVec, &faceAuxArray);
    }

    PetscScalar* faceSolutionGradArray;
    PetscScalar* faceAuxGradArray;
    VecGetArray(faceSolutionGradVec, &faceSolutionGradArray);
    if (auxTotalSize) {
        VecGetArray(faceAuxGradVec, &faceAuxGradArray);
    }

    PetscInt iFace = 0;
    for (PetscInt face = globalFaceStart; face < fEnd; face++) {
        auto& stencil = stencils[iFace];

        if (!stencil.stencilSize) {
            continue;
        }

        // get the field faces
        PetscScalar* faceSolValues;
        DMPlexPointLocalRead(faceSolutionDm, face, faceSolutionArray, &faceSolValues) >> checkError;
        PetscArrayzero(faceSolValues, solTotalSize) >> checkError;
        PetscScalar* faceAuxValues;
        if (auxTotalSize) {
            DMPlexPointLocalRead(faceAuxDm, face, faceAuxArray, &faceAuxValues) >> checkError;
            PetscArrayzero(faceAuxValues, auxTotalSize) >> checkError;
        }

        // compute the value on the face
        for (PetscInt c = 0; c < stencil.stencilSize; c++) {
            PetscInt cell = stencil.stencil[c];

            // get cell value and add to the array
            PetscScalar* solutionValue;
            DMPlexPointLocalRead(solutionDm, cell, solutionArray, &solutionValue) >> checkError;
            AddToArray(solTotalSize, solutionValue, faceSolValues, stencil.weights[c]);

            if (auxTotalSize) {
                // get cell value and add to the array
                PetscScalar* auxValue;
                DMPlexPointLocalRead(auxDm, cell, auxArray, &auxValue) >> checkError;
                AddToArray(auxTotalSize, auxValue, faceAuxValues, stencil.weights[c]);
            }
        }

        // get the grad faces
        PetscScalar* faceSolGradValues;
        DMPlexPointLocalRead(faceSolutionGradDm, face, faceSolutionGradArray, &faceSolGradValues) >> checkError;
        PetscArrayzero(faceSolGradValues, solTotalSize * dim) >> checkError;
        PetscScalar* faceAuxGradValues;
        if (auxTotalSize) {
            DMPlexPointLocalRead(faceAuxGradDm, face, faceAuxGradArray, &faceAuxGradValues) >> checkError;
            PetscArrayzero(faceAuxGradValues, auxTotalSize * dim) >> checkError;
        }

        // Using this value compute the gradient on the faces
        for (PetscInt c = 0; c < stencil.stencilSize; c++) {
            PetscInt cell = stencil.stencil[c];

            // get cell value and add to the array
            PetscScalar* solutionValue;
            DMPlexPointLocalRead(solutionDm, cell, solutionArray, &solutionValue) >> checkError;

            // for each component
            PetscInt offset = 0;
            for (PetscInt cc = 0; cc < solTotalSize; cc++) {
                PetscScalar delta = solutionValue[cc] - faceSolValues[cc];

                for (PetscInt d = 0; d < dim; ++d) {
                    faceSolGradValues[offset++] += stencil.gradientWeights[cc * dim + d] * delta;
                }
            }

            if (auxTotalSize) {
                // get cell value and add to the array
                PetscScalar* auxValue;
                DMPlexPointLocalRead(auxDm, cell, auxArray, &auxValue) >> checkError;

                // for each component
                offset = 0;
                for (PetscInt cc = 0; cc < auxTotalSize; cc++) {
                    PetscScalar delta = auxValue[cc] - faceAuxValues[cc];

                    for (PetscInt d = 0; d < dim; ++d) {
                        faceAuxGradValues[offset++] += stencil.gradientWeights[cc * dim + d] * delta;
                    }
                }
            }
        }
    }

    // clean up the array
    VecRestoreArrayRead(solutionVec, &solutionArray);
    VecRestoreArray(faceSolutionVec, &faceSolutionArray);
    VecRestoreArray(faceSolutionGradVec, &faceSolutionGradArray);

    if (auxTotalSize) {
        VecRestoreArrayRead(auxVec, &auxArray);
        VecRestoreArray(faceAuxVec, &faceAuxArray);
        VecRestoreArray(faceAuxGradVec, &faceAuxGradArray);
    }
}
void ablate::finiteVolume::FaceInterpolant::RestoreInterpolatedFaceVectors(Vec solutionVec, Vec auxVec, Vec& faceSolutionVec, Vec& faceAuxVec, Vec& faceSolutionGradVec, Vec& faceAuxGradVec) {
    // Size the return vectors
    DMRestoreLocalVector(faceSolutionDm, &faceSolutionVec) >> checkError;
    DMRestoreLocalVector(faceSolutionGradDm, &faceAuxVec) >> checkError;
    if (faceAuxDm) {
        DMRestoreLocalVector(faceAuxDm, &faceSolutionGradVec) >> checkError;
    }
    if (faceAuxGradDm) {
        DMRestoreLocalVector(faceAuxGradDm, &faceAuxGradVec) >> checkError;
    }
}
void ablate::finiteVolume::FaceInterpolant::ComputeRHS(PetscReal time, Vec locXVec, Vec locAuxVec, Vec locFVec, std::vector<FaceInterpolant::ContinuousFluxFunctionDescription>& rhsFunctions,
                                                       PetscInt fStart, PetscInt fEnd, const PetscInt* faces, Vec cellGeomVec) {
    // get the dm
    auto dm = subDomain->GetDM();

    // interpolate to the faces
    Vec faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec;
    GetInterpolatedFaceVectors(locXVec, locAuxVec, faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec);

    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;

    // extract the arrays for each of the vec
    const PetscScalar* faceSolutionArray;
    const PetscScalar* faceAuxArray;
    VecGetArrayRead(faceSolutionVec, &faceSolutionArray);
    if (auxTotalSize) {
        VecGetArrayRead(faceAuxVec, &faceAuxArray);
    }

    const PetscScalar* faceSolutionGradArray;
    const PetscScalar* faceAuxGradArray;
    VecGetArrayRead(faceSolutionGradVec, &faceSolutionGradArray);
    if (auxTotalSize) {
        VecGetArrayRead(faceAuxGradVec, &faceAuxGradArray);
    }

    // get raw access to the locF
    PetscScalar* locFArray;
    VecGetArray(locFVec, &locFArray) >> checkError;

    // Size up the total dim
    PetscInt totDim;
    PetscDSGetTotalDimension(subDomain->GetDiscreteSystem(), &totDim) >> checkError;
    std::vector<PetscScalar> flux(totDim);
    auto dim = subDomain->GetDimensions();

    // Get the geometry for the mesh
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM) >> checkError;
    const PetscScalar* cellGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    // march only over this region
    DMLabel regionLabel;
    PetscInt regionValue;
    ablate::domain::Region::GetLabel(region, subDomain->GetDM(), regionLabel, regionValue);

    // Precompute the offsets to pass into the rhsFluxFunctionDescriptions
    std::vector<PetscInt> fluxComponentSize(rhsFunctions.size());
    std::vector<PetscInt> fluxId(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> uOff(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> aOff(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> uOff_x(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> aOff_x(rhsFunctions.size());

    // Get the full set of offsets from the ds
    PetscInt* uOffTotal;
    PetscInt* uGradOffTotal;
    PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &uOffTotal) >> checkError;
    PetscDSGetComponentDerivativeOffsets(subDomain->GetDiscreteSystem(), &uGradOffTotal) >> checkError;

    for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
        const auto& field = subDomain->GetField(rhsFunctions[fun].field);
        fluxComponentSize[fun] = field.numberComponents;
        fluxId[fun] = field.id;
        for (std::size_t f = 0; f < rhsFunctions[fun].inputFields.size(); f++) {
            uOff[fun].push_back(uOffTotal[rhsFunctions[fun].inputFields[f]]);
            uOff_x[fun].push_back(uGradOffTotal[rhsFunctions[fun].inputFields[f]]);
        }
    }

    if (auxTotalSize) {
        PetscInt* auxOffTotal;
        PetscInt* auxGradOffTotal;
        PetscDSGetComponentOffsets(subDomain->GetAuxDiscreteSystem(), &auxOffTotal) >> checkError;
        PetscDSGetComponentDerivativeOffsets(subDomain->GetAuxDiscreteSystem(), &auxGradOffTotal) >> checkError;
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            for (std::size_t f = 0; f < rhsFunctions[fun].auxFields.size(); f++) {
                aOff[fun].push_back(auxOffTotal[rhsFunctions[fun].auxFields[f]]);
                aOff_x[fun].push_back(auxGradOffTotal[rhsFunctions[fun].auxFields[f]]);
            }
        }
    }

    // march over each face
    for (PetscInt f = fStart; f < fEnd; f++) {
        PetscInt face = faces ? faces[f] : f;

        // extract the stencil for this face
        const auto& stencil = stencils[face - globalFaceStart];

        // make sure that this is a valid face
        PetscInt ghost, nsupp, nchild;
        DMLabelGetValue(ghostLabel, face, &ghost) >> checkError;
        DMPlexGetSupportSize(subDomain->GetDM(), face, &nsupp) >> checkError;
        DMPlexGetTreeChildren(subDomain->GetDM(), face, &nchild, nullptr) >> checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

        // extract the arrays
        const PetscScalar* solutionValue;
        DMPlexPointLocalRead(faceSolutionDm, face, faceSolutionArray, &solutionValue) >> checkError;
        const PetscScalar* solutionGradValue;
        DMPlexPointLocalRead(faceSolutionGradDm, face, faceSolutionGradArray, &solutionGradValue) >> checkError;

        const PetscScalar* auxValue = nullptr;
        const PetscScalar* auxGradValue = nullptr;
        if (auxTotalSize) {
            DMPlexPointLocalRead(faceAuxDm, face, faceAuxArray, &auxValue) >> checkError;
            DMPlexPointLocalRead(faceAuxGradDm, face, faceAuxGradArray, &auxGradValue) >> checkError;
        }

        // determine where to add the cell values
        const PetscInt* faceCells;
        PetscFVCellGeom *cgL, *cgR;
        DMPlexGetSupport(subDomain->GetDM(), face, &faceCells) >> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[0], cellGeomArray, &cgL) >> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[1], cellGeomArray, &cgR) >> checkError;

        // March over each source function
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            PetscArrayzero(flux.data(), totDim) >> checkError;

            const auto& rhsFluxFunctionDescription = rhsFunctions[fun];
            rhsFluxFunctionDescription.function(dim,
                                                stencil.area,
                                                stencil.normal,
                                                stencil.centroid,
                                                uOff[fun].data(),
                                                uOff_x[fun].data(),
                                                solutionValue,
                                                solutionGradValue,
                                                aOff[fun].data(),
                                                aOff_x[fun].data(),
                                                auxValue,
                                                auxGradValue,
                                                flux.data(),
                                                rhsFluxFunctionDescription.context) >>
                checkError;

            // add the flux back to the cell
            PetscScalar *fL = nullptr, *fR = nullptr;
            PetscInt cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[0], rhsFunctions[fun].field, locFArray, &fL) >> checkError;
            }

            cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[1], rhsFunctions[fun].field, locFArray, &fR) >> checkError;
            }

            for (PetscInt d = 0; d < fluxComponentSize[fun]; ++d) {
                if (fL) fL[d] -= flux[d] / cgL->volume;
                if (fR) fR[d] += flux[d] / cgR->volume;
            }
        }
    }

    VecRestoreArrayRead(faceSolutionVec, &faceSolutionArray);
    VecRestoreArrayRead(faceSolutionGradVec, &faceSolutionGradArray);

    if (auxTotalSize) {
        VecRestoreArrayRead(faceSolutionGradVec, &faceSolutionGradArray);
        VecRestoreArrayRead(faceAuxGradVec, &faceAuxGradArray);
    }
    VecRestoreArray(locFVec, &locFArray) >> checkError;

    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    RestoreInterpolatedFaceVectors(locXVec, locFVec, faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec);
}

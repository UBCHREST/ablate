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

    // Determine the max sizes for scratch arrays
    PetscInt maxConeSize, maxSupportSize;
    DMPlexGetMaxSizes(subDomain->GetDM(), &maxConeSize, &maxSupportSize) >> checkError;
    std::vector<PetscInt> faceNodes((1 + maxConeSize) * 2);
    PetscInt* faceNodesPointer = faceNodes.data();
    std::vector<PetscInt> nodeCells((1 + maxSupportSize) * 2);
    PetscInt* nodeCellsPointer = nodeCells.data();

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
        // Get all nodes in this face
        PetscInt numberNodes;
        DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &numberNodes, &faceNodesPointer) >> checkError;

        // For each node get the cells that connect to it
        for (PetscInt n = 0; n < numberNodes; n++) {
            PetscInt numberCells;
            DMPlexGetTransitiveClosure(dm, faceNodes[n * 2], PETSC_FALSE, &numberCells, &nodeCellsPointer) >> checkError;

            for (PetscInt c = 0; c < numberCells; c++) {
                PetscInt cell = nodeCells[c * 2];
                // Make sure that cell is in this region and is a cell
                PetscInt cellHeight;
                DMPlexGetPointHeight(dm, cell, &cellHeight) >> checkError;
                if (cellHeight != 0) {
                    continue;
                }
                if (regionLabel) {
                    PetscInt labelValue;
                    DMLabelGetValue(regionLabel, cell, &labelValue) >> checkError;
                    if (labelValue != regionValue) {
                        continue;
                    }
                }

                stencil.stencil.push_back(cell);
            }
        }

        // Clean up the stencil to remove duplicates
        std::sort(stencil.stencil.begin(), stencil.stencil.end());
        stencil.stencil.erase(std::unique(stencil.stencil.begin(), stencil.stencil.end()), stencils[iFace].stencil.end());

        // ignore cell if there are no stencils
        if (stencil.stencilSize) {
            // for now, set the interpolant weights to be the average of the two faces
            stencil.stencilSize = (PetscInt)stencil.stencil.size();
            stencil.stencil.resize(stencil.stencilSize, 0.0);
            stencil.stencil.resize(stencil.stencilSize * subDomain->GetDimensions(), 0.0);
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

            // Add this geometry to the BoundaryFVFaceGeom
            PetscFVFaceGeom* fg;
            DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;
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
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(subDomain->GetDM(), 1, &fStart, &fEnd) >> checkError;
    stencils.resize(fEnd - fStart);

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
    for (PetscInt face = fStart; face < fEnd; face++) {
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
            for (PetscInt cc = 0; cc < solTotalSize; c++) {
                PetscScalar delta = solutionValue[cc] - faceSolValues[cc];

                for (PetscInt d = 0; d < dim; ++d) {
                    faceSolGradValues[offset++] += stencil.gradientWeights[c * dim + d] * delta;
                }
            }

            if (auxTotalSize) {
                // get cell value and add to the array
                PetscScalar* auxValue;
                DMPlexPointLocalRead(auxDm, cell, auxArray, &auxValue) >> checkError;

                // for each component
                offset = 0;
                for (PetscInt cc = 0; cc < auxTotalSize; c++) {
                    PetscScalar delta = auxValue[cc] - faceAuxValues[cc];

                    for (PetscInt d = 0; d < dim; ++d) {
                        faceAuxGradValues[offset++] += stencil.gradientWeights[c * dim + d] * delta;
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
        VecRestoreArray(faceSolutionGradVec, &faceSolutionGradArray);
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

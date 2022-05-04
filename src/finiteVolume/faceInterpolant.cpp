#include "faceInterpolant.hpp"
#include "finiteVolume/stencils/faceStencilGenerator.hpp"
#include "finiteVolume/stencils/leastSquares.hpp"
#include "finiteVolume/stencils/leastSquaresAverage.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::FaceInterpolant::FaceInterpolant(const std::shared_ptr<ablate::domain::SubDomain>& subDomain, Vec faceGeomVec, Vec cellGeomVec) : subDomain(subDomain) {
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
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(subDomain->GetDM(), 1, &fStart, &fEnd) >> checkError;
    stencils.resize(fEnd - fStart);

    // extract the dm
    auto dm = subDomain->GetDM();

    // Set up the gradient calculator
    std::unique_ptr<stencil::FaceStencilGenerator> faceStencilGenerator;
    if (subDomain->GetDimensions() == 1) {
        faceStencilGenerator = std::make_unique<stencil::LeastSquares>();
    } else {
        faceStencilGenerator = std::make_unique<stencil::LeastSquaresAverage>();
    }

    // Get the geometry for the mesh
    DM faceDM, cellDM;
    VecGetDM(faceGeomVec, &faceDM) >> checkError;
    VecGetDM(cellGeomVec, &cellDM) >> checkError;
    const PetscScalar* cellGeomArray;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    // perform some fv and mpi ghost cell checks
    PetscInt gcStart;
    DMPlexGetGhostCellStratum(dm, &gcStart, nullptr) >> checkError;

    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> checkError;

    // Compute the stencil for each face
    PetscInt iFace = 0;
    for (PetscInt face = fStart; face < fEnd; face++) {
        auto& stencil = stencils[iFace++];

        // make sure that this is a valid face
        PetscInt ghost, nsupp, nchild;
        DMLabelGetValue(ghostLabel, face, &ghost) >> checkError;
        DMPlexGetSupportSize(subDomain->GetDM(), face, &nsupp) >> checkError;
        DMPlexGetTreeChildren(subDomain->GetDM(), face, &nchild, nullptr) >> checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

        faceStencilGenerator->Generate(face, stencil, *subDomain, cellDM, cellGeomArray, faceDM, faceGeomArray);
    }

    // clean up the geom
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
    DMGetLocalVector(faceSolutionGradDm, &faceSolutionGradVec) >> checkError;
    if (auxTotalSize) {
        DMGetLocalVector(faceAuxDm, &faceAuxVec) >> checkError;
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

    PetscInt iFacee = 0;
    for (PetscInt face = globalFaceStart; face < fEnd; face++) {
        auto& stencil = stencils[iFacee];
        iFacee++;
        if (!stencil.stencilSize) {
            PetscScalar* faceSolValues;
            DMPlexPointLocalRead(faceSolutionDm, face, faceSolutionArray, &faceSolValues) >> checkError;
            utilities::MathUtilities::ScaleVector(solTotalSize, faceSolValues, (double)NAN);
            PetscScalar* faceAuxValues;
            if (auxTotalSize) {
                DMPlexPointLocalRead(faceAuxDm, face, faceAuxArray, &faceAuxValues) >> checkError;
                utilities::MathUtilities::ScaleVector(auxTotalSize, faceAuxValues, (double)NAN);
            }
            PetscScalar* faceSolGradValues;
            DMPlexPointLocalRead(faceSolutionGradDm, face, faceSolutionGradArray, &faceSolGradValues) >> checkError;
            utilities::MathUtilities::ScaleVector(solTotalSize * dim, faceSolGradValues, (double)NAN);
            PetscScalar* faceAuxGradValues;
            if (auxTotalSize) {
                DMPlexPointLocalRead(faceAuxGradDm, face, faceAuxGradArray, &faceAuxGradValues) >> checkError;
                utilities::MathUtilities::ScaleVector(auxTotalSize * dim, faceAuxGradValues, (double)NAN);
            }

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

            // compute the value on the face
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

            // for each component compute the gradient
            PetscInt offset = 0;
            for (PetscInt cc = 0; cc < solTotalSize; cc++) {
                for (PetscInt d = 0; d < dim; ++d) {
                    faceSolGradValues[offset++] += stencil.gradientWeights[c * dim + d] * solutionValue[cc];
                }
            }

            if (auxTotalSize) {
                // get cell value and add to the array
                PetscScalar* auxValue;
                DMPlexPointLocalRead(auxDm, cell, auxArray, &auxValue) >> checkError;

                // for each component
                offset = 0;
                for (PetscInt cc = 0; cc < auxTotalSize; cc++) {
                    for (PetscInt d = 0; d < dim; ++d) {
                        faceAuxGradValues[offset++] += stencil.gradientWeights[c * dim + d] * auxValue[cc];
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

void ablate::finiteVolume::FaceInterpolant::RestoreInterpolatedFaceVectors(Vec, Vec, Vec& faceSolutionVec, Vec& faceAuxVec, Vec& faceSolutionGradVec, Vec& faceAuxGradVec) {
    // Size the return vectors
    DMRestoreLocalVector(faceSolutionDm, &faceSolutionVec) >> checkError;
    DMRestoreLocalVector(faceSolutionGradDm, &faceSolutionGradVec) >> checkError;
    if (faceAuxDm) {
        DMRestoreLocalVector(faceAuxDm, &faceAuxVec) >> checkError;
    }
    if (faceAuxGradDm) {
        DMRestoreLocalVector(faceAuxGradDm, &faceAuxGradVec) >> checkError;
    }
}
void ablate::finiteVolume::FaceInterpolant::ComputeRHS(PetscReal time, Vec locXVec, Vec locAuxVec, Vec locFVec, const std::shared_ptr<domain::Region>& solverRegion,
                                                       std::vector<FaceInterpolant::ContinuousFluxFunctionDescription>& rhsFunctions, const solver::Range& faceRange, Vec cellGeomVec,
                                                       Vec faceGeomVec) {
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
    DM faceDM;
    VecGetDM(faceGeomVec, &faceDM) >> checkError;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;

    // march only over this region
    DMLabel regionLabel;
    PetscInt regionValue;
    ablate::domain::Region::GetLabel(solverRegion, subDomain->GetDM(), regionLabel, regionValue);

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
    for (PetscInt f = faceRange.start; f < faceRange.end; f++) {
        PetscInt face = faceRange.points ? faceRange.points[f] : f;

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

        PetscFVFaceGeom* fg;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg);

        // March over each source function
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            PetscArrayzero(flux.data(), totDim) >> checkError;

            const auto& rhsFluxFunctionDescription = rhsFunctions[fun];
            rhsFluxFunctionDescription.function(dim,
                                                fg,
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
        VecRestoreArrayRead(faceAuxVec, &faceAuxArray);
        VecRestoreArrayRead(faceAuxGradVec, &faceAuxGradArray);
    }
    VecRestoreArray(locFVec, &locFArray) >> checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
    RestoreInterpolatedFaceVectors(locXVec, locAuxVec, faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec);
}

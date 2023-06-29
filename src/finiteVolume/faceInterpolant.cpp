#include "faceInterpolant.hpp"
#include "finiteVolume/stencils/faceStencilGenerator.hpp"
#include "finiteVolume/stencils/leastSquares.hpp"
#include "finiteVolume/stencils/leastSquaresAverage.hpp"
#include "utilities/mathUtilities.hpp"

ablate::finiteVolume::FaceInterpolant::FaceInterpolant(const std::shared_ptr<ablate::domain::SubDomain>& subDomain, const std::shared_ptr<domain::Region> solverRegion, Vec faceGeomVec,
                                                       Vec cellGeomVec)
    : subDomain(subDomain) {
    auto ds = subDomain->GetDiscreteSystem();
    PetscDSGetTotalDimension(ds, &solTotalSize) >> utilities::PetscUtilities::checkError;
    CreateFaceDm(solTotalSize, subDomain->GetDM(), faceSolutionDm);
    CreateFaceDm(solTotalSize * subDomain->GetDimensions(), subDomain->GetDM(), faceSolutionGradDm);

    auto auxDs = subDomain->GetAuxDiscreteSystem();
    if (auxDs) {
        PetscDSGetTotalDimension(auxDs, &auxTotalSize) >> utilities::PetscUtilities::checkError;
        CreateFaceDm(auxTotalSize, subDomain->GetDM(), faceAuxDm);
        CreateFaceDm(auxTotalSize * subDomain->GetDimensions(), subDomain->GetDM(), faceAuxGradDm);
    }

    // Size up the stencil
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(subDomain->GetDM(), 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;
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
    VecGetDM(faceGeomVec, &faceDM) >> utilities::PetscUtilities::checkError;
    VecGetDM(cellGeomVec, &cellDM) >> utilities::PetscUtilities::checkError;
    const PetscScalar* cellGeomArray;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

    // perform some fv and mpi ghost cell checks
    PetscInt gcStart;
    DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &gcStart, nullptr) >> utilities::PetscUtilities::checkError;

    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    // Compute the stencil for each face
    PetscInt iFace = 0;
    for (PetscInt face = fStart; face < fEnd; face++) {
        auto& stencil = stencils[iFace++];

        // make sure that this is a valid face
        PetscInt ghost, nsupp, nchild;
        DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupportSize(subDomain->GetDM(), face, &nsupp) >> utilities::PetscUtilities::checkError;
        DMPlexGetTreeChildren(subDomain->GetDM(), face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

        faceStencilGenerator->Generate(face, stencil, *subDomain, solverRegion, cellDM, cellGeomArray, faceDM, faceGeomArray);
    }

    // clean up the geom
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
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
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;

    // create a face solution dm for that is the required number of variables per face
    PetscSection solutionSection;
    DMClone(dm, &newDm) >> utilities::PetscUtilities::checkError;
    PetscSectionCreate(PetscObjectComm((PetscObject)dm), &solutionSection) >> utilities::PetscUtilities::checkError;

    PetscSectionSetChart(solutionSection, fStart, fEnd) >> utilities::PetscUtilities::checkError;
    for (PetscInt f = fStart; f < fEnd; ++f) {
        PetscSectionSetDof(solutionSection, f, totalDim) >> utilities::PetscUtilities::checkError;
    }

    PetscSectionSetUp(solutionSection) >> utilities::PetscUtilities::checkError;
    DMSetLocalSection(newDm, solutionSection) >> utilities::PetscUtilities::checkError;
    PetscSectionDestroy(&solutionSection) >> utilities::PetscUtilities::checkError;
}

void ablate::finiteVolume::FaceInterpolant::GetInterpolatedFaceVectors(Vec solutionVec, Vec auxVec, Vec& faceSolutionVec, Vec& faceAuxVec, Vec& faceSolutionGradVec, Vec& faceAuxGradVec) {
    // Compute the stencil for each face
    auto dim = subDomain->GetDimensions();
    PetscInt fEnd;
    DMPlexGetHeightStratum(subDomain->GetDM(), 1, &globalFaceStart, &fEnd) >> utilities::PetscUtilities::checkError;
    stencils.resize(fEnd - globalFaceStart);

    // Size the return vectors
    DMGetLocalVector(faceSolutionDm, &faceSolutionVec) >> utilities::PetscUtilities::checkError;
    DMGetLocalVector(faceSolutionGradDm, &faceSolutionGradVec) >> utilities::PetscUtilities::checkError;
    if (auxTotalSize) {
        DMGetLocalVector(faceAuxDm, &faceAuxVec) >> utilities::PetscUtilities::checkError;
        DMGetLocalVector(faceAuxGradDm, &faceAuxGradVec) >> utilities::PetscUtilities::checkError;
    }

    // Extract each of the dms needed
    DM solutionDm, auxDm;
    VecGetDM(solutionVec, &solutionDm) >> utilities::PetscUtilities::checkError;
    if (auxVec) {
        VecGetDM(auxVec, &auxDm) >> utilities::PetscUtilities::checkError;
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
            DMPlexPointLocalRead(faceSolutionDm, face, faceSolutionArray, &faceSolValues) >> utilities::PetscUtilities::checkError;
            utilities::MathUtilities::ScaleVector(solTotalSize, faceSolValues, (double)NAN);
            PetscScalar* faceAuxValues;
            if (auxTotalSize) {
                DMPlexPointLocalRead(faceAuxDm, face, faceAuxArray, &faceAuxValues) >> utilities::PetscUtilities::checkError;
                utilities::MathUtilities::ScaleVector(auxTotalSize, faceAuxValues, (double)NAN);
            }
            PetscScalar* faceSolGradValues;
            DMPlexPointLocalRead(faceSolutionGradDm, face, faceSolutionGradArray, &faceSolGradValues) >> utilities::PetscUtilities::checkError;
            utilities::MathUtilities::ScaleVector(solTotalSize * dim, faceSolGradValues, (double)NAN);
            PetscScalar* faceAuxGradValues;
            if (auxTotalSize) {
                DMPlexPointLocalRead(faceAuxGradDm, face, faceAuxGradArray, &faceAuxGradValues) >> utilities::PetscUtilities::checkError;
                utilities::MathUtilities::ScaleVector(auxTotalSize * dim, faceAuxGradValues, (double)NAN);
            }

            continue;
        }

        // get the field faces
        PetscScalar* faceSolValues;
        DMPlexPointLocalRead(faceSolutionDm, face, faceSolutionArray, &faceSolValues) >> utilities::PetscUtilities::checkError;
        PetscArrayzero(faceSolValues, solTotalSize) >> utilities::PetscUtilities::checkError;
        PetscScalar* faceAuxValues;
        if (auxTotalSize) {
            DMPlexPointLocalRead(faceAuxDm, face, faceAuxArray, &faceAuxValues) >> utilities::PetscUtilities::checkError;
            PetscArrayzero(faceAuxValues, auxTotalSize) >> utilities::PetscUtilities::checkError;
        }

        // get the grad faces
        PetscScalar* faceSolGradValues;
        DMPlexPointLocalRead(faceSolutionGradDm, face, faceSolutionGradArray, &faceSolGradValues) >> utilities::PetscUtilities::checkError;
        PetscArrayzero(faceSolGradValues, solTotalSize * dim) >> utilities::PetscUtilities::checkError;
        PetscScalar* faceAuxGradValues;
        if (auxTotalSize) {
            DMPlexPointLocalRead(faceAuxGradDm, face, faceAuxGradArray, &faceAuxGradValues) >> utilities::PetscUtilities::checkError;
            PetscArrayzero(faceAuxGradValues, auxTotalSize * dim) >> utilities::PetscUtilities::checkError;
        }

        // Using this value compute the gradient on the faces
        for (PetscInt c = 0; c < stencil.stencilSize; c++) {
            PetscInt cell = stencil.stencil[c];

            // compute the value on the face
            // get cell value and add to the array
            PetscScalar* solutionValue;
            DMPlexPointLocalRead(solutionDm, cell, solutionArray, &solutionValue) >> utilities::PetscUtilities::checkError;
            AddToArray(solTotalSize, solutionValue, faceSolValues, stencil.weights[c]);

            if (auxTotalSize) {
                // get cell value and add to the array
                PetscScalar* auxValue;
                DMPlexPointLocalRead(auxDm, cell, auxArray, &auxValue) >> utilities::PetscUtilities::checkError;
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
                DMPlexPointLocalRead(auxDm, cell, auxArray, &auxValue) >> utilities::PetscUtilities::checkError;

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
    DMRestoreLocalVector(faceSolutionDm, &faceSolutionVec) >> utilities::PetscUtilities::checkError;
    DMRestoreLocalVector(faceSolutionGradDm, &faceSolutionGradVec) >> utilities::PetscUtilities::checkError;
    if (faceAuxDm) {
        DMRestoreLocalVector(faceAuxDm, &faceAuxVec) >> utilities::PetscUtilities::checkError;
    }
    if (faceAuxGradDm) {
        DMRestoreLocalVector(faceAuxGradDm, &faceAuxGradVec) >> utilities::PetscUtilities::checkError;
    }
}
void ablate::finiteVolume::FaceInterpolant::ComputeRHS(PetscReal time, Vec locXVec, Vec locAuxVec, Vec locFVec, const std::shared_ptr<domain::Region>& solverRegion,
                                                       std::vector<FaceInterpolant::ContinuousFluxFunctionDescription>& rhsFunctions, const ablate::domain::Range& faceRange, Vec cellGeomVec,
                                                       Vec faceGeomVec) {
    // get the dm
    auto dm = subDomain->GetDM();

    // interpolate to the faces
    Vec faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec;
    GetInterpolatedFaceVectors(locXVec, locAuxVec, faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec);

    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(subDomain->GetDM(), "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

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
    VecGetArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;

    // Size up the total dim
    PetscInt totDim;
    PetscDSGetTotalDimension(subDomain->GetDiscreteSystem(), &totDim) >> utilities::PetscUtilities::checkError;
    std::vector<PetscScalar> flux(totDim);
    auto dim = subDomain->GetDimensions();

    // Get the geometry for the mesh
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM) >> utilities::PetscUtilities::checkError;
    const PetscScalar* cellGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    DM faceDM;
    VecGetDM(faceGeomVec, &faceDM) >> utilities::PetscUtilities::checkError;
    const PetscScalar* faceGeomArray;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;

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
    PetscDSGetComponentOffsets(subDomain->GetDiscreteSystem(), &uOffTotal) >> utilities::PetscUtilities::checkError;
    PetscDSGetComponentDerivativeOffsets(subDomain->GetDiscreteSystem(), &uGradOffTotal) >> utilities::PetscUtilities::checkError;

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
        PetscDSGetComponentOffsets(subDomain->GetAuxDiscreteSystem(), &auxOffTotal) >> utilities::PetscUtilities::checkError;
        PetscDSGetComponentDerivativeOffsets(subDomain->GetAuxDiscreteSystem(), &auxGradOffTotal) >> utilities::PetscUtilities::checkError;
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
        DMLabelGetValue(ghostLabel, face, &ghost) >> utilities::PetscUtilities::checkError;
        DMPlexGetSupportSize(subDomain->GetDM(), face, &nsupp) >> utilities::PetscUtilities::checkError;
        DMPlexGetTreeChildren(subDomain->GetDM(), face, &nchild, nullptr) >> utilities::PetscUtilities::checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

        // extract the arrays
        const PetscScalar* solutionValue;
        DMPlexPointLocalRead(faceSolutionDm, face, faceSolutionArray, &solutionValue) >> utilities::PetscUtilities::checkError;
        const PetscScalar* solutionGradValue;
        DMPlexPointLocalRead(faceSolutionGradDm, face, faceSolutionGradArray, &solutionGradValue) >> utilities::PetscUtilities::checkError;

        const PetscScalar* auxValue = nullptr;
        const PetscScalar* auxGradValue = nullptr;
        if (auxTotalSize) {
            DMPlexPointLocalRead(faceAuxDm, face, faceAuxArray, &auxValue) >> utilities::PetscUtilities::checkError;
            DMPlexPointLocalRead(faceAuxGradDm, face, faceAuxGradArray, &auxGradValue) >> utilities::PetscUtilities::checkError;
        }

        // determine where to add the cell values
        const PetscInt* faceCells;
        PetscFVCellGeom *cgL, *cgR;
        DMPlexGetSupport(subDomain->GetDM(), face, &faceCells) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRead(cellDM, faceCells[0], cellGeomArray, &cgL) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalRead(cellDM, faceCells[1], cellGeomArray, &cgR) >> utilities::PetscUtilities::checkError;

        PetscFVFaceGeom* fg;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg);

        // March over each source function
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            PetscArrayzero(flux.data(), totDim) >> utilities::PetscUtilities::checkError;

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
                utilities::PetscUtilities::checkError;

            // add the flux back to the cell
            PetscScalar *fL = nullptr, *fR = nullptr;
            PetscInt cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> utilities::PetscUtilities::checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> utilities::PetscUtilities::checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[0], rhsFunctions[fun].field, locFArray, &fL) >> utilities::PetscUtilities::checkError;
            }

            cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> utilities::PetscUtilities::checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> utilities::PetscUtilities::checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[1], rhsFunctions[fun].field, locFArray, &fR) >> utilities::PetscUtilities::checkError;
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
    VecRestoreArray(locFVec, &locFArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    RestoreInterpolatedFaceVectors(locXVec, locAuxVec, faceSolutionVec, faceAuxVec, faceSolutionGradVec, faceAuxGradVec);
}

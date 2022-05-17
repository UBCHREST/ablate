#include "cellInterpolant.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utility>

ablate::finiteVolume::CellInterpolant::CellInterpolant(std::shared_ptr<ablate::domain::SubDomain> subDomainIn, const std::shared_ptr<domain::Region>& solverRegion, Vec faceGeomVec, Vec cellGeomVec)
    : subDomain(std::move(std::move(subDomainIn))) {
    auto getGradientDm = [this, solverRegion, faceGeomVec, cellGeomVec](const domain::Field& fieldInfo, std::vector<DM>& gradDMs) {
        auto petscField = subDomain->GetPetscFieldObject(fieldInfo);
        auto petscFieldFV = (PetscFV)petscField;

        PetscBool computeGradients;
        PetscFVGetComputeGradients(petscFieldFV, &computeGradients) >> checkError;

        if (computeGradients) {
            DM dmGradInt;

            DMLabel regionLabel = nullptr;
            PetscInt regionValue = PETSC_DECIDE;
            domain::Region::GetLabel(solverRegion, subDomain->GetDM(), regionLabel, regionValue);

            ComputeGradientFVM(subDomain->GetFieldDM(fieldInfo), regionLabel, regionValue, petscFieldFV, faceGeomVec, cellGeomVec, &dmGradInt) >> checkError;
            gradDMs.push_back(dmGradInt);
        } else {
            gradDMs.push_back(nullptr);
        }
    };

    // Compute the gradient dm for each field that supports it
    for (const auto& fieldInfo : subDomain->GetFields()) {
        getGradientDm(fieldInfo, gradientCellDms);
    }
}

ablate::finiteVolume::CellInterpolant::~CellInterpolant() {
    for (auto& dm : gradientCellDms) {
        if (dm) {
            DMDestroy(&dm) >> checkError;
        }
    }
}

void ablate::finiteVolume::CellInterpolant::ComputeRHS(PetscReal time, Vec locXVec, Vec locAuxVec, Vec locFVec, const std::shared_ptr<domain::Region>& solverRegion,
                                                       std::vector<CellInterpolant::DiscontinuousFluxFunctionDescription>& rhsFunctions, const solver::Range& faceRange, const solver::Range& cellRange,
                                                       Vec cellGeomVec, Vec faceGeomVec) {
    auto dm = subDomain->GetDM();
    auto dmAux = subDomain->GetAuxDM();

    /* 1: Get sizes from dm and dmAux */
    PetscSection section = nullptr;
    DMGetLocalSection(dm, &section) >> checkError;

    // Get the ds from he subDomain and required info
    auto ds = subDomain->GetDiscreteSystem();
    PetscInt nf, totDim;
    PetscDSGetNumFields(ds, &nf) >> checkError;
    PetscDSGetTotalDimension(ds, &totDim) >> checkError;

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    PetscDS dsAux = subDomain->GetAuxDiscreteSystem();
    PetscInt naf = 0, totDimAux = 0;
    if (locAuxVec) {
        PetscDSGetTotalDimension(dsAux, &totDimAux) >> checkError;
        PetscDSGetNumFields(dsAux, &naf) >> checkError;
    }

    /* 2: Get geometric data */
    // We can use a single call for the geometry data because it does not depend on the fv object
    const PetscScalar* cellGeomArray = nullptr;
    const PetscScalar* faceGeomArray = nullptr;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
    DM faceDM, cellDM;
    VecGetDM(faceGeomVec, &faceDM) >> checkError;
    VecGetDM(cellGeomVec, &cellDM) >> checkError;

    // Get raw access to the computed values
    const PetscScalar *xArray, *auxArray = nullptr;
    VecGetArrayRead(locXVec, &xArray) >> checkError;
    if (locAuxVec) {
        VecGetArrayRead(locAuxVec, &auxArray) >> checkError;
    }

    // get raw access to the locF
    PetscScalar* locFArray;
    VecGetArray(locFVec, &locFArray) >> checkError;

    // there must be a separate gradient vector/dm for field because they can be different sizes
    std::vector<Vec> locGradVecs(nf, nullptr);

    /* Reconstruct and limit cell gradients */
    // for each field compute the gradient in the localGrads vector
    for (const auto& field : subDomain->GetFields()) {
        ComputeFieldGradients(field, locXVec, locGradVecs[field.subId], gradientCellDms[field.subId], cellGeomVec, faceGeomVec, faceRange, cellRange);
    }

    std::vector<const PetscScalar*> locGradArrays(nf, nullptr);
    for (const auto& field : subDomain->GetFields()) {
        if (locGradVecs[field.subId]) {
            VecGetArrayRead(locGradVecs[field.subId], &locGradArrays[field.subId]) >> checkError;
        }
    }

    ComputeFluxSourceTerms(dm,
                           ds,
                           totDim,
                           xArray,
                           dmAux,
                           dsAux,
                           totDimAux,
                           auxArray,
                           faceDM,
                           faceGeomArray,
                           cellDM,
                           cellGeomArray,
                           gradientCellDms,
                           locGradArrays,
                           locFArray,
                           solverRegion,
                           rhsFunctions,
                           faceRange,
                           cellRange);

    // clean up cell grads
    for (const auto& field : subDomain->GetFields()) {
        if (locGradVecs[field.subId]) {
            VecRestoreArrayRead(locGradVecs[field.subId], &locGradArrays[field.subId]) >> checkError;
            DMRestoreLocalVector(gradientCellDms[field.subId], &locGradVecs[field.subId]) >> checkError;
        }
    }

    // cleanup (restore access to locGradVecs, locAuxGradVecs with DMRestoreLocalVector)
    VecRestoreArrayRead(locXVec, &xArray) >> checkError;
    if (locAuxVec) {
        VecRestoreArrayRead(locAuxVec, &auxArray) >> checkError;
    }

    VecRestoreArray(locFVec, &locFArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, (const PetscScalar**)&faceGeomArray) >> checkError;
    VecRestoreArrayRead(cellGeomVec, (const PetscScalar**)&cellGeomArray) >> checkError;
}

void ablate::finiteVolume::CellInterpolant::ComputeRHS(PetscReal time, Vec locXVec, Vec locAuxVec, Vec locFVec, const std::shared_ptr<domain::Region>& solverRegion,
                                                       std::vector<CellInterpolant::PointFunctionDescription>& rhsFunctions, const solver::Range& cellRange, Vec cellGeomVec) {
    auto dm = subDomain->GetDM();
    auto dmAux = subDomain->GetAuxDM();

    /* 1: Get sizes from dm and dmAux */
    PetscSection section = nullptr;
    DMGetLocalSection(dm, &section) >> checkError;

    // Get the ds from he subDomain and required info
    auto ds = subDomain->GetDiscreteSystem();
    PetscInt nf, totDim;
    PetscDSGetNumFields(ds, &nf) >> checkError;
    PetscDSGetTotalDimension(ds, &totDim) >> checkError;

    // Check to see if the dm has an auxVec/auxDM associated with it.  If it does, extract it
    PetscDS dsAux = subDomain->GetAuxDiscreteSystem();
    PetscInt naf = 0, totDimAux = 0;
    if (locAuxVec) {
        PetscDSGetTotalDimension(dsAux, &totDimAux) >> checkError;
        PetscDSGetNumFields(dsAux, &naf) >> checkError;
    }

    // We can use a single call for the geometry data because it does not depend on the fv object
    const PetscScalar* cellGeomArray = nullptr;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    DM cellDM;
    VecGetDM(cellGeomVec, &cellDM) >> checkError;

    // Get raw access to the computed values
    const PetscScalar *xArray, *auxArray = nullptr;
    VecGetArrayRead(locXVec, &xArray) >> checkError;
    if (locAuxVec) {
        VecGetArrayRead(locAuxVec, &auxArray) >> checkError;
    }

    // get raw access to the locF
    PetscScalar* locFArray;
    VecGetArray(locFVec, &locFArray) >> checkError;

    // Compute the source terms from flux across the interface for cell based gradient functions
    // Precompute the offsets to pass into the rhsFluxFunctionDescriptions
    std::vector<std::vector<PetscInt>> fluxComponentSize(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> fluxComponentOffset(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> uOff(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> aOff(rhsFunctions.size());

    // Get the full set of offsets from the ds
    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(ds, &uOffTotal) >> checkError;

    for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
        for (std::size_t f = 0; f < rhsFunctions[fun].fields.size(); f++) {
            const auto& field = subDomain->GetField(rhsFunctions[fun].fields[f]);

            PetscInt fieldSize, fieldOffset;
            PetscDSGetFieldSize(ds, field.subId, &fieldSize) >> checkError;
            PetscDSGetFieldOffset(ds, field.subId, &fieldOffset) >> checkError;
            fluxComponentSize[fun].push_back(fieldSize);
            fluxComponentOffset[fun].push_back(fieldOffset);
        }

        for (std::size_t f = 0; f < rhsFunctions[fun].inputFields.size(); f++) {
            uOff[fun].push_back(uOffTotal[rhsFunctions[fun].inputFields[f]]);
        }
    }

    if (dsAux) {
        PetscInt* auxOffTotal;
        PetscDSGetComponentOffsets(dsAux, &auxOffTotal) >> checkError;
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            for (std::size_t f = 0; f < rhsFunctions[fun].auxFields.size(); f++) {
                aOff[fun].push_back(auxOffTotal[rhsFunctions[fun].auxFields[f]]);
            }
        }
    }

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

    PetscInt dim = subDomain->GetDimensions();

    // Size up a scratch variable
    PetscScalar* fScratch;
    PetscCalloc1(totDim, &fScratch);

    // March over each cell
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // make sure that this is not a ghost cell
        if (ghostLabel) {
            PetscInt ghostVal;

            DMLabelGetValue(ghostLabel, cell, &ghostVal) >> checkError;
            if (ghostVal > 0) continue;
        }

        // extract the point locations for this cell
        const PetscFVCellGeom* cg;
        const PetscScalar* u;
        PetscScalar* rhs;
        DMPlexPointLocalRead(cellDM, cell, cellGeomArray, &cg) >> checkError;
        DMPlexPointLocalRead(dm, cell, xArray, &u) >> checkError;
        DMPlexPointLocalRef(dm, cell, locFArray, &rhs) >> checkError;

        // if there is an aux field, get it
        const PetscScalar* a = nullptr;
        if (auxArray) {
            DMPlexPointLocalRead(dmAux, cell, auxArray, &a) >> checkError;
        }

        // March over each functionDescriptions
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            rhsFunctions[fun].function(dim, time, cg, uOff[fun].data(), u, aOff[fun].data(), a, fScratch, rhsFunctions[fun].context) >> checkError;

            // copy over each result flux field
            PetscInt r = 0;
            for (std::size_t ff = 0; ff < rhsFunctions[fun].fields.size(); ff++) {
                for (PetscInt d = 0; d < fluxComponentSize[fun][ff]; ++d) {
                    rhs[fluxComponentOffset[fun][ff] + d] += fScratch[r++];
                }
            }
        }
    }

    // cleanup (restore access to locGradVecs, locAuxGradVecs with DMRestoreLocalVector)
    VecRestoreArrayRead(locXVec, &xArray) >> checkError;
    if (locAuxVec) {
        VecRestoreArrayRead(locAuxVec, &auxArray) >> checkError;
    }

    VecRestoreArray(locFVec, &locFArray) >> checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
}

/**
 * This is a duplication of PETSC that we don't have access to
 */
static PetscErrorCode DMPlexApplyLimiter_Internal(DM dm, DM dmCell, PetscLimiter lim, PetscInt dim, PetscInt dof, PetscInt cell, PetscInt field, PetscInt face, PetscInt fStart, PetscInt fEnd,
                                                  PetscReal* cellPhi, const PetscScalar* x, const PetscScalar* cellgeom, const PetscFVCellGeom* cg, const PetscScalar* cx, const PetscScalar* cgrad) {
    const PetscInt* children;
    PetscInt numChildren;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = DMPlexGetTreeChildren(dm, face, &numChildren, &children);
    CHKERRQ(ierr);
    if (numChildren) {
        PetscInt c;

        for (c = 0; c < numChildren; c++) {
            PetscInt childFace = children[c];

            if (childFace >= fStart && childFace < fEnd) {
                ierr = DMPlexApplyLimiter_Internal(dm, dmCell, lim, dim, dof, cell, field, childFace, fStart, fEnd, cellPhi, x, cellgeom, cg, cx, cgrad);
                CHKERRQ(ierr);
            }
        }
    } else {
        PetscScalar* ncx;
        PetscFVCellGeom* ncg;
        const PetscInt* fcells;
        PetscInt ncell, d;
        PetscReal v[3];

        ierr = DMPlexGetSupport(dm, face, &fcells);
        CHKERRQ(ierr);
        ncell = cell == fcells[0] ? fcells[1] : fcells[0];
        if (field >= 0) {
            ierr = DMPlexPointLocalFieldRead(dm, ncell, field, x, &ncx);
            CHKERRQ(ierr);
        } else {
            ierr = DMPlexPointLocalRead(dm, ncell, x, &ncx);
            CHKERRQ(ierr);
        }
        ierr = DMPlexPointLocalRead(dmCell, ncell, cellgeom, &ncg);
        CHKERRQ(ierr);
        DMPlex_WaxpyD_Internal(dim, -1, cg->centroid, ncg->centroid, v);
        for (d = 0; d < dof; ++d) {
            /* We use the symmetric slope limited form of Berger, Aftosmis, and Murman 2005 */
            PetscReal denom = DMPlex_DotD_Internal(dim, &cgrad[d * dim], v);
            PetscReal phi, flim = 0.5 * PetscRealPart(ncx[d] - cx[d]) / denom;

            ierr = PetscLimiterLimit(lim, flim, &phi);
            CHKERRQ(ierr);
            cellPhi[d] = PetscMin(cellPhi[d], phi);
        }
    }
    PetscFunctionReturn(0);
}

void ablate::finiteVolume::CellInterpolant::ComputeFieldGradients(const domain::Field& field, Vec xLocalVec, Vec& gradLocVec, DM& dmGrad, Vec cellGeomVec, Vec faceGeomVec,
                                                                  const solver::Range& faceRange, const solver::Range& cellRange) {
    // get the FVM petsc field associated with this field
    auto fvm = (PetscFV)subDomain->GetPetscFieldObject(field);
    auto dm = subDomain->GetFieldDM(field);

    // Get the dm for this grad field
    // If there is no grad, return
    if (!dmGrad) {
        return;
    }

    // Create a gradLocVec
    DMGetLocalVector(dmGrad, &gradLocVec) >> checkError;

    // Get the correct sized vec (gradient for this field)
    Vec gradGlobVec;
    DMGetGlobalVector(dmGrad, &gradGlobVec) >> checkError;
    VecZeroEntries(gradGlobVec) >> checkError;

    // check to see if there is a ghost label
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

    // Get the face geometry
    DM dmFace;
    const PetscScalar* faceGeometryArray;
    VecGetDM(faceGeomVec, &dmFace) >> checkError;
    VecGetArrayRead(faceGeomVec, &faceGeometryArray);

    // extract the local x array
    const PetscScalar* xLocalArray;
    VecGetArrayRead(xLocalVec, &xLocalArray);

    // extract the global grad array
    PetscScalar* gradGlobArray;
    VecGetArray(gradGlobVec, &gradGlobArray);

    // Get the dof and dim
    PetscInt dim = subDomain->GetDimensions();
    PetscInt dof = field.numberComponents;

    for (PetscInt f = faceRange.start; f < faceRange.end; ++f) {
        PetscInt face = faceRange.points ? faceRange.points[f] : f;

        // make sure that this is a face we should use
        PetscBool boundary;
        PetscInt ghost = -1;
        if (ghostLabel) {
            DMLabelGetValue(ghostLabel, face, &ghost);
        }
        DMIsBoundaryPoint(dm, face, &boundary);
        PetscInt numChildren;
        DMPlexGetTreeChildren(dm, face, &numChildren, nullptr);
        if (ghost >= 0 || boundary || numChildren) continue;

        // Do a sanity check on the number of cells connected to this face
        PetscInt numCells;
        DMPlexGetSupportSize(dm, face, &numCells);
        if (numCells != 2) {
            throw std::runtime_error("face " + std::to_string(face) + " has " + std::to_string(numCells) + " support points (cells): expected 2");
        }

        // add in the contributions from this face
        const PetscInt* cells;
        PetscFVFaceGeom* fg;
        PetscScalar* cx[2];
        PetscScalar* cgrad[2];

        DMPlexGetSupport(dm, face, &cells);
        DMPlexPointLocalRead(dmFace, face, faceGeometryArray, &fg);
        for (PetscInt c = 0; c < 2; ++c) {
            DMPlexPointLocalFieldRead(dm, cells[c], field.id, xLocalArray, &cx[c]) >> checkError;
            DMPlexPointGlobalRef(dmGrad, cells[c], gradGlobArray, &cgrad[c]) >> checkError;
        }
        for (PetscInt pd = 0; pd < dof; ++pd) {
            PetscScalar delta = cx[1][pd] - cx[0][pd];

            for (PetscInt d = 0; d < dim; ++d) {
                if (cgrad[0]) cgrad[0][pd * dim + d] += fg->grad[0][d] * delta;
                if (cgrad[1]) cgrad[1][pd * dim + d] -= fg->grad[1][d] * delta;
            }
        }
    }

    // Check for a limiter the limiter
    PetscLimiter lim;
    PetscFVGetLimiter(fvm, &lim) >> checkError;
    if (lim) {
        /* Limit interior gradients (using cell-based loop because it generalizes better to vector limiters) */
        // Get the cell geometry
        DM dmCell;
        const PetscScalar* cellGeometryArray;
        VecGetDM(cellGeomVec, &dmCell) >> checkError;
        VecGetArrayRead(cellGeomVec, &cellGeometryArray);

        // create a temp work array
        PetscReal* cellPhi;
        DMGetWorkArray(dm, dof, MPIU_REAL, &cellPhi) >> checkError;

        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            PetscInt cell = cellRange.points ? cellRange.points[c] : c;

            const PetscInt* cellFaces;
            PetscScalar* cx;
            PetscFVCellGeom* cg;
            PetscScalar* cgrad;
            PetscInt coneSize;

            DMPlexGetConeSize(dm, cell, &coneSize) >> checkError;
            DMPlexGetCone(dm, cell, &cellFaces) >> checkError;
            DMPlexPointLocalFieldRead(dm, cell, field.id, xLocalArray, &cx) >> checkError;
            DMPlexPointLocalRead(dmCell, cell, cellGeometryArray, &cg) >> checkError;
            DMPlexPointGlobalRef(dmGrad, cell, gradGlobArray, &cgrad) >> checkError;

            if (!cgrad) {
                /* Unowned overlap cell, we do not compute */
                continue;
            }
            /* Limiter will be minimum value over all neighbors */
            for (PetscInt d = 0; d < dof; ++d) {
                cellPhi[d] = PETSC_MAX_REAL;
            }
            for (PetscInt f = 0; f < coneSize; ++f) {
                DMPlexApplyLimiter_Internal(dm, dmCell, lim, dim, dof, cell, field.id, cellFaces[f], faceRange.start, faceRange.end, cellPhi, xLocalArray, cellGeometryArray, cg, cx, cgrad) >>
                    checkError;
            }
            /* Apply limiter to gradient */
            for (PetscInt pd = 0; pd < dof; ++pd) {
                /* Scalar limiter applied to each component separately */
                for (PetscInt d = 0; d < dim; ++d) {
                    cgrad[pd * dim + d] *= cellPhi[pd];
                }
            }
        }

        // clean up the limiter work
        DMRestoreWorkArray(dm, dof, MPIU_REAL, &cellPhi) >> checkError;
        VecRestoreArrayRead(cellGeomVec, &cellGeometryArray);
    }
    // Communicate gradient values
    VecRestoreArray(gradGlobVec, &gradGlobArray) >> checkError;
    DMGlobalToLocalBegin(dmGrad, gradGlobVec, INSERT_VALUES, gradLocVec) >> checkError;
    DMGlobalToLocalEnd(dmGrad, gradGlobVec, INSERT_VALUES, gradLocVec) >> checkError;

    // cleanup
    VecRestoreArrayRead(xLocalVec, &xLocalArray) >> checkError;
    VecRestoreArrayRead(faceGeomVec, &faceGeometryArray) >> checkError;
    DMRestoreGlobalVector(dmGrad, &gradGlobVec) >> checkError;
}

void ablate::finiteVolume::CellInterpolant::ComputeFluxSourceTerms(DM dm, PetscDS ds, PetscInt totDim, const PetscScalar* xArray, DM dmAux, PetscDS dsAux, PetscInt totDimAux,
                                                                   const PetscScalar* auxArray, DM faceDM, const PetscScalar* faceGeomArray, DM cellDM, const PetscScalar* cellGeomArray,
                                                                   std::vector<DM>& dmGrads, std::vector<const PetscScalar*>& locGradArrays, PetscScalar* locFArray,
                                                                   const std::shared_ptr<domain::Region>& solverRegion,
                                                                   std::vector<CellInterpolant::DiscontinuousFluxFunctionDescription>& rhsFunctions, const solver::Range& faceRange,
                                                                   const solver::Range& cellRange) {
    PetscInt dim = subDomain->GetDimensions();

    // Size up the work arrays (uL, uR, gradL, gradR, auxL, auxR, gradAuxL, gradAuxR), these are only sized for one face at a time
    PetscScalar* flux;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &flux) >> checkError;

    PetscScalar *uL, *uR;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &uL) >> checkError;
    DMGetWorkArray(dm, totDim, MPIU_SCALAR, &uR) >> checkError;

    PetscScalar *gradL, *gradR;
    DMGetWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradL) >> checkError;
    DMGetWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradR) >> checkError;

    // size up the aux variables
    PetscScalar *auxL = nullptr, *auxR = nullptr;

    // Precompute the offsets to pass into the rhsFluxFunctionDescriptions
    std::vector<PetscInt> fluxComponentSize(rhsFunctions.size());
    std::vector<PetscInt> fluxId(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> uOff(rhsFunctions.size());
    std::vector<std::vector<PetscInt>> aOff(rhsFunctions.size());

    // Get the full set of offsets from the ds
    PetscInt* uOffTotal;
    PetscDSGetComponentOffsets(ds, &uOffTotal) >> checkError;

    for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
        const auto& field = subDomain->GetField(rhsFunctions[fun].field);
        fluxComponentSize[fun] = field.numberComponents;
        fluxId[fun] = field.id;
        for (std::size_t f = 0; f < rhsFunctions[fun].inputFields.size(); f++) {
            uOff[fun].push_back(uOffTotal[rhsFunctions[fun].inputFields[f]]);
        }
    }

    if (dsAux) {
        PetscInt* auxOffTotal;
        PetscDSGetComponentOffsets(dsAux, &auxOffTotal) >> checkError;
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            for (std::size_t f = 0; f < rhsFunctions[fun].auxFields.size(); f++) {
                aOff[fun].push_back(auxOffTotal[rhsFunctions[fun].auxFields[f]]);
            }
        }
    }
    // check for ghost cells
    DMLabel ghostLabel;
    DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

    // get the label for this region
    DMLabel regionLabel = nullptr;
    PetscInt regionValue = 0;
    domain::Region::GetLabel(solverRegion, subDomain->GetDM(), regionLabel, regionValue);

    // March over each face in this region
    for (PetscInt f = faceRange.start; f < faceRange.end; ++f) {
        const PetscInt face = faceRange.points ? faceRange.points[f] : f;

        // make sure that this is a valid face
        PetscInt ghost, nsupp, nchild;
        DMLabelGetValue(ghostLabel, face, &ghost) >> checkError;
        DMPlexGetSupportSize(dm, face, &nsupp) >> checkError;
        DMPlexGetTreeChildren(dm, face, &nchild, nullptr) >> checkError;
        if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;

        // Get the face geometry
        const PetscInt* faceCells;
        PetscFVFaceGeom* fg;
        PetscFVCellGeom *cgL, *cgR;
        DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;
        DMPlexGetSupport(dm, face, &faceCells) >> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[0], cellGeomArray, &cgL) >> checkError;
        DMPlexPointLocalRead(cellDM, faceCells[1], cellGeomArray, &cgR) >> checkError;

        PetscInt leftFlowLabelValue = regionValue;
        PetscInt rightFlowLabelValue = regionValue;
        if (regionLabel) {
            DMLabelGetValue(regionLabel, faceCells[0], &leftFlowLabelValue);
            DMLabelGetValue(regionLabel, faceCells[1], &rightFlowLabelValue);
        }
        // compute the left/right face values
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[0], *cgL, dm, xArray, dmGrads, locGradArrays, uL, gradL, leftFlowLabelValue == regionValue);
        ProjectToFace(subDomain->GetFields(), ds, *fg, faceCells[1], *cgR, dm, xArray, dmGrads, locGradArrays, uR, gradR, rightFlowLabelValue == regionValue);

        // determine the left/right cells
        if (auxArray) {
            // Get the field values at this cell
            DMPlexPointLocalRead(dmAux, faceCells[0], auxArray, &auxL) >> checkError;
            DMPlexPointLocalRead(dmAux, faceCells[1], auxArray, &auxR) >> checkError;
        }

        // March over each source function
        for (std::size_t fun = 0; fun < rhsFunctions.size(); fun++) {
            PetscArrayzero(flux, totDim) >> checkError;
            const auto& rhsFluxFunctionDescription = rhsFunctions[fun];
            rhsFluxFunctionDescription.function(dim, fg, uOff[fun].data(), uL, uR, aOff[fun].data(), auxL, auxR, flux, rhsFluxFunctionDescription.context) >> checkError;

            // add the flux back to the cell
            PetscScalar *fL = nullptr, *fR = nullptr;
            PetscInt cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[0], &ghost) >> checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[0], &cellLabelValue) >> checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[0], fluxId[fun], locFArray, &fL) >> checkError;
            }

            cellLabelValue = regionValue;
            DMLabelGetValue(ghostLabel, faceCells[1], &ghost) >> checkError;
            if (regionLabel) {
                DMLabelGetValue(regionLabel, faceCells[1], &cellLabelValue) >> checkError;
            }
            if (ghost <= 0 && regionValue == cellLabelValue) {
                DMPlexPointLocalFieldRef(dm, faceCells[1], fluxId[fun], locFArray, &fR) >> checkError;
            }

            for (PetscInt d = 0; d < fluxComponentSize[fun]; ++d) {
                if (fL) fL[d] -= flux[d] / cgL->volume;
                if (fR) fR[d] += flux[d] / cgR->volume;
            }
        }
    }

    // cleanup
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &flux) >> checkError;
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &uL) >> checkError;
    DMRestoreWorkArray(dm, totDim, MPIU_SCALAR, &uR) >> checkError;
    DMRestoreWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradL) >> checkError;
    DMRestoreWorkArray(dm, dim * totDim, MPIU_SCALAR, &gradR) >> checkError;
}

static PetscErrorCode BuildGradientReconstruction_Internal(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, DM dmFace, PetscScalar* fgeom, DM dmCell, PetscScalar* cgeom) {
    DMLabel ghostLabel;
    PetscScalar *dx, *grad, **gref;
    PetscInt dim, cStart, cEnd, c, cEndInterior, maxNumFaces;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, nullptr));
    cEndInterior = cEndInterior < 0 ? cEnd : cEndInterior;
    PetscCall(DMPlexGetMaxSizes(dm, &maxNumFaces, nullptr));
    PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
    PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
    PetscCall(PetscMalloc3(maxNumFaces * dim, &dx, maxNumFaces * dim, &grad, maxNumFaces, &gref));
    for (c = cStart; c < cEndInterior; c++) {
        const PetscInt* faces;
        PetscInt numFaces, usedFaces, f, d;
        PetscFVCellGeom* cg;
        PetscBool boundary;
        PetscInt ghost;
        PetscInt labelValue;

        // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
        PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
        if (ghost >= 0) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
        PetscCall(DMPlexGetConeSize(dm, c, &numFaces));
        PetscCall(DMPlexGetCone(dm, c, &faces));
        PetscCheck(!(numFaces < dim), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %" PetscInt_FMT " has only %" PetscInt_FMT " faces, not enough for gradient reconstruction", c, numFaces);
        for (f = 0, usedFaces = 0; f < numFaces; ++f) {
            PetscFVCellGeom* cg1;
            PetscFVFaceGeom* fg;
            const PetscInt* fcells;
            PetscInt ncell, side;

            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, faces[f], &labelValue));
                if (labelValue != regionValue) continue;
            }

            PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
            PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
            if ((ghost >= 0) || boundary) continue;
            PetscCall(DMPlexGetSupport(dm, faces[f], &fcells));
            side = (c != fcells[0]); /* c is on left=0 or right=1 of face */
            ncell = fcells[!side];   /* the neighbor */
            PetscCall(DMPlexPointLocalRef(dmFace, faces[f], fgeom, &fg));
            PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
            for (d = 0; d < dim; ++d) dx[usedFaces * dim + d] = cg1->centroid[d] - cg->centroid[d];
            gref[usedFaces++] = fg->grad[side]; /* Gradient reconstruction term will go here */
        }
        PetscCheck(usedFaces, PETSC_COMM_SELF, PETSC_ERR_USER, "Mesh contains isolated cell (no neighbors). Is it intentional?");
        PetscCall(PetscFVComputeGradient(fvm, usedFaces, dx, grad));
        for (f = 0, usedFaces = 0; f < numFaces; ++f) {
            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, faces[f], &labelValue));
                if (labelValue != regionValue) continue;
            }
            PetscCall(DMLabelGetValue(ghostLabel, faces[f], &ghost));
            PetscCall(DMIsBoundaryPoint(dm, faces[f], &boundary));
            if ((ghost >= 0) || boundary) continue;
            for (d = 0; d < dim; ++d) gref[usedFaces][d] = grad[usedFaces * dim + d];
            ++usedFaces;
        }
    }
    PetscCall(PetscFree3(dx, grad, gref));
    PetscFunctionReturn(0);
}

static PetscErrorCode BuildGradientReconstruction_Internal_Tree(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, DM dmFace, PetscScalar* fgeom, DM dmCell, PetscScalar* cgeom) {
    DMLabel ghostLabel;
    PetscScalar *dx, *grad, **gref;
    PetscInt dim, cStart, cEnd, c, cEndInterior, fStart, fEnd, f, nStart, nEnd, maxNumFaces = 0;
    PetscSection neighSec;
    PetscInt(*neighbors)[2];
    PetscInt* counter;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, nullptr));
    if (cEndInterior < 0) cEndInterior = cEnd;
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &neighSec));
    PetscCall(PetscSectionSetChart(neighSec, cStart, cEndInterior));
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(DMGetLabel(dm, "ghost", &ghostLabel));
    for (f = fStart; f < fEnd; f++) {
        const PetscInt* fcells;
        PetscBool boundary;
        PetscInt ghost = -1;
        PetscInt numChildren, numCells, labelValue;

        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
        PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
        PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, nullptr));
        if ((ghost >= 0) || boundary || numChildren) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
        if (numCells == 2) {
            PetscCall(DMPlexGetSupport(dm, f, &fcells));
            for (c = 0; c < 2; c++) {
                PetscInt cell = fcells[c];

                if (cell >= cStart && cell < cEndInterior) {
                    PetscCall(PetscSectionAddDof(neighSec, cell, 1));
                }
            }
        }
    }
    PetscCall(PetscSectionSetUp(neighSec));
    PetscCall(PetscSectionGetMaxDof(neighSec, &maxNumFaces));
    PetscCall(PetscFVLeastSquaresSetMaxFaces(fvm, maxNumFaces));
    nStart = 0;
    PetscCall(PetscSectionGetStorageSize(neighSec, &nEnd));
    PetscCall(PetscMalloc1((nEnd - nStart), &neighbors));
    PetscCall(PetscCalloc1((cEndInterior - cStart), &counter));
    for (f = fStart; f < fEnd; f++) {
        const PetscInt* fcells;
        PetscBool boundary;
        PetscInt ghost = -1;
        PetscInt numChildren, numCells, labelValue;

        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, f, &ghost));
        PetscCall(DMIsBoundaryPoint(dm, f, &boundary));
        PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, nullptr));
        if ((ghost >= 0) || boundary || numChildren) continue;

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
            if (labelValue != regionValue) continue;
        }

        PetscCall(DMPlexGetSupportSize(dm, f, &numCells));
        if (numCells == 2) {
            PetscCall(DMPlexGetSupport(dm, f, &fcells));
            for (c = 0; c < 2; c++) {
                PetscInt cell = fcells[c], off;

                if (regionLabel) {
                    PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
                    if (labelValue != regionValue) continue;
                }

                if (cell >= cStart && cell < cEndInterior) {
                    PetscCall(PetscSectionGetOffset(neighSec, cell, &off));
                    off += counter[cell - cStart]++;
                    neighbors[off][0] = f;
                    neighbors[off][1] = fcells[1 - c];
                }
            }
        }
    }
    PetscCall(PetscFree(counter));
    PetscCall(PetscMalloc3(maxNumFaces * dim, &dx, maxNumFaces * dim, &grad, maxNumFaces, &gref));
    for (c = cStart; c < cEndInterior; c++) {
        PetscInt numFaces, d, off, labelValue, ghost = -1;
        PetscFVCellGeom* cg;

        PetscCall(DMPlexPointLocalRead(dmCell, c, cgeom, &cg));
        PetscCall(PetscSectionGetDof(neighSec, c, &numFaces));
        PetscCall(PetscSectionGetOffset(neighSec, c, &off));

        if (regionLabel) {
            PetscCall(DMLabelGetValue(regionLabel, c, &labelValue));
            if (labelValue != regionValue) continue;
        }

        // do not attempt to compute a gradient reconstruction stencil in a ghost cell.  It will never be used
        if (ghostLabel) PetscCall(DMLabelGetValue(ghostLabel, c, &ghost));
        if (ghost >= 0) continue;

        PetscCheck(!(numFaces < dim), PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %" PetscInt_FMT " has only %" PetscInt_FMT " faces, not enough for gradient reconstruction", c, numFaces);
        for (f = 0; f < numFaces; ++f) {
            PetscFVCellGeom* cg1;
            PetscFVFaceGeom* fg;
            const PetscInt* fcells;
            PetscInt ncell, side, nface;

            if (regionLabel) {
                PetscCall(DMLabelGetValue(regionLabel, f, &labelValue));
                if (labelValue != regionValue) continue;
            }

            nface = neighbors[off + f][0];
            ncell = neighbors[off + f][1];
            PetscCall(DMPlexGetSupport(dm, nface, &fcells));
            side = (c != fcells[0]);
            PetscCall(DMPlexPointLocalRef(dmFace, nface, fgeom, &fg));
            PetscCall(DMPlexPointLocalRead(dmCell, ncell, cgeom, &cg1));
            for (d = 0; d < dim; ++d) dx[f * dim + d] = cg1->centroid[d] - cg->centroid[d];
            gref[f] = fg->grad[side]; /* Gradient reconstruction term will go here */
        }
        PetscCall(PetscFVComputeGradient(fvm, numFaces, dx, grad));
        for (f = 0; f < numFaces; ++f) {
            for (d = 0; d < dim; ++d) gref[f][d] = grad[f * dim + d];
        }
    }
    PetscCall(PetscFree3(dx, grad, gref));
    PetscCall(PetscSectionDestroy(&neighSec));
    PetscCall(PetscFree(neighbors));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::CellInterpolant::ComputeGradientFVM(DM dm, DMLabel regionLabel, PetscInt regionValue, PetscFV fvm, Vec faceGeometry, Vec cellGeometry, DM* dmGrad) {
    DM dmFace, dmCell;
    PetscScalar *fgeom, *cgeom;
    PetscSection sectionGrad, parentSection;
    PetscInt dim, pdim, cStart, cEnd, cEndInterior, c;

    PetscFunctionBegin;
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(PetscFVGetNumComponents(fvm, &pdim));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetGhostCellStratum(dm, &cEndInterior, nullptr));
    /* Construct the interpolant corresponding to each face from the least-square solution over the cell neighborhood */
    PetscCall(VecGetDM(faceGeometry, &dmFace));
    PetscCall(VecGetDM(cellGeometry, &dmCell));
    PetscCall(VecGetArray(faceGeometry, &fgeom));
    PetscCall(VecGetArray(cellGeometry, &cgeom));
    PetscCall(DMPlexGetTree(dm, &parentSection, nullptr, nullptr, nullptr, nullptr));
    if (!parentSection) {
        PetscCall(BuildGradientReconstruction_Internal(dm, regionLabel, regionValue, fvm, dmFace, fgeom, dmCell, cgeom));
    } else {
        PetscCall(BuildGradientReconstruction_Internal_Tree(dm, regionLabel, regionValue, fvm, dmFace, fgeom, dmCell, cgeom));
    }
    PetscCall(VecRestoreArray(faceGeometry, &fgeom));
    PetscCall(VecRestoreArray(cellGeometry, &cgeom));
    /* Create storage for gradients */
    PetscCall(DMClone(dm, dmGrad));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionGrad));
    PetscCall(PetscSectionSetChart(sectionGrad, cStart, cEnd));
    for (c = cStart; c < cEnd; ++c) PetscCall(PetscSectionSetDof(sectionGrad, c, pdim * dim));
    PetscCall(PetscSectionSetUp(sectionGrad));
    PetscCall(DMSetLocalSection(*dmGrad, sectionGrad));
    PetscCall(PetscSectionDestroy(&sectionGrad));
    PetscFunctionReturn(0);
}
void ablate::finiteVolume::CellInterpolant::ProjectToFace(const std::vector<domain::Field>& fields, PetscDS ds, const PetscFVFaceGeom& faceGeom, PetscInt cellId, const PetscFVCellGeom& cellGeom,
                                                          DM dm, const PetscScalar* xArray, const std::vector<DM>& dmGrads, const std::vector<const PetscScalar*>& gradArrays, PetscScalar* u,
                                                          PetscScalar* grad, bool projectField) {
    const auto dim = subDomain->GetDimensions();

    // Keep track of derivative offset
    PetscInt* offsets;
    PetscInt* dirOffsets;
    PetscDSGetComponentOffsets(ds, &offsets) >> checkError;
    PetscDSGetComponentDerivativeOffsets(ds, &dirOffsets) >> checkError;

    // March over each field
    for (const auto& field : fields) {
        PetscReal dx[3];
        PetscScalar* xCell;
        PetscScalar* gradCell;

        // Get the field values at this cell
        DMPlexPointLocalFieldRead(dm, cellId, field.subId, xArray, &xCell) >> checkError;

        // If we need to project the field
        if (projectField && dmGrads[field.subId]) {
            DMPlexPointLocalRead(dmGrads[field.subId], cellId, gradArrays[field.subId], &gradCell) >> checkError;
            DMPlex_WaxpyD_Internal(dim, -1, cellGeom.centroid, faceGeom.centroid, dx);

            // Project the cell centered value onto the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c] + DMPlex_DotD_Internal(dim, &gradCell[c * dim], dx);

                // copy the gradient into the grad vector
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = gradCell[c * dim + d];
                }
            }

        } else if (dmGrads[field.subId]) {
            // Project the cell centered value onto the face
            DMPlexPointLocalRead(dmGrads[field.subId], cellId, gradArrays[field.subId], &gradCell) >> checkError;
            // Project the cell centered value onto the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c];

                // copy the gradient into the grad vector
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = gradCell[c * dim + d];
                }
            }

        } else {
            // Just copy the cell centered value on to the face
            for (PetscInt c = 0; c < field.numberComponents; ++c) {
                u[offsets[field.subId] + c] = xCell[c];

                // fill the grad with NAN to prevent use
                for (PetscInt d = 0; d < dim; d++) {
                    grad[dirOffsets[field.subId] + c * dim + d] = NAN;
                }
            }
        }
    }
}

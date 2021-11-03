#include "ghost.hpp"
#include <petsc.h>
#include <utilities/petscError.hpp>

ablate::finiteVolume::boundaryConditions::Ghost::Ghost(std::string fieldName, std::string boundaryName, std::vector<int> labelIds,
                                                       ablate::finiteVolume::boundaryConditions::Ghost::UpdateFunction updateFunction, void *updateContext, std::string labelNameIn)
    : BoundaryCondition(boundaryName, fieldName),
      labelName(labelNameIn.empty() ? "Face Sets" : labelNameIn),
      labelIds(labelIds.begin(), labelIds.end()),
      updateFunction(updateFunction),
      updateContext(updateContext) {}

ablate::finiteVolume::boundaryConditions::Ghost::Ghost(std::string fieldName, std::string boundaryName, int labelId, ablate::finiteVolume::boundaryConditions::Ghost::UpdateFunction updateFunction,
                                                       void *updateContext, std::string labelName)
    : Ghost(fieldName, boundaryName, std::vector<int>{labelId}, updateFunction, updateContext, labelName) {}

void ablate::finiteVolume::boundaryConditions::Ghost::SetupBoundary(DM dm, PetscDS problem, PetscInt fieldId) {
    DMLabel label;
    DMGetLabel(dm, labelName.c_str(), &label) >> checkError;
    PetscDSAddBoundary(problem, DM_BC_NATURAL_RIEMANN, GetBoundaryName().c_str(), label, labelIds.size(), &labelIds[0], fieldId, 0, NULL, (void(*)(void))NULL, NULL, (void *)NULL, NULL) >> checkError;

    // extract some information about the flowField
    PetscDSGetFieldSize(problem, fieldId, &fieldSize) >> checkError;
    PetscDSGetCoordinateDimension(problem, &dim) >> checkError;
    PetscDSGetFieldOffset(problem, fieldId, &fieldOffset) >> checkError;
}

void ablate::finiteVolume::boundaryConditions::Ghost::InsertBoundaryValues(ablate::domain::SubDomain &subDomain, PetscReal time, Vec faceGeometryVec, Vec cellGeometryVec, Vec locXVec) {
    // extract some information about the flowField
    auto &fieldId = subDomain.GetSolutionField(GetFieldName());

    // get mesh information
    auto dm = subDomain.GetDM();
    PetscSF sf;
    const PetscInt *leaves;
    PetscInt nleaves;
    DMGetPointSF(dm, &sf) >> checkError;
    PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL) >> checkError;
    nleaves = PetscMax(0, nleaves);
    DMGetDimension(dm, &dim) >> checkError;

    // get all faces in this dm
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> checkError;

    // get the face/cells geometry and dms
    DM dmFace;
    const PetscScalar *faceGeometryArray;
    VecGetDM(faceGeometryVec, &dmFace) >> checkError;
    VecGetArrayRead(faceGeometryVec, &faceGeometryArray) >> checkError;

    DM dmCell;
    const PetscScalar *cellGeometryArray;
    VecGetDM(cellGeometryVec, &dmCell) >> checkError;
    VecGetArrayRead(cellGeometryVec, &cellGeometryArray) >> checkError;

    // Get the locX array
    PetscScalar *xArray;
    VecGetArray(locXVec, &xArray) >> checkError;

    DMLabel label;
    DMGetLabel(dm, labelName.c_str(), &label) >> checkError;

    // get the FVLabel
    auto fvLabel = subDomain.GetLabel();

    // for each id in this label
    for (const auto &id : labelIds) {
        IS faceIS;
        const PetscInt *faces;
        PetscInt numFaces, f;

        DMLabelGetStratumIS(label, id, &faceIS) >> checkError;
        if (!faceIS) continue; /* No points with that id on this process */
        ISGetLocalSize(faceIS, &numFaces) >> checkError;
        ISGetIndices(faceIS, &faces) >> checkError;
        for (f = 0; f < numFaces; ++f) {
            const PetscInt face = faces[f], *cells;
            PetscFVFaceGeom *fg;

            if ((face < fStart) || (face >= fEnd)) continue; /* Refinement adds non-faces to labels */
            PetscInt loc;
            PetscFindInt(face, nleaves, (PetscInt *)leaves, &loc) >> checkError;
            if (loc >= 0) continue;
            DMPlexPointLocalRead(dmFace, face, faceGeometryArray, &fg) >> checkError;
            DMPlexGetSupport(dm, face, &cells) >> checkError;

            // make sure we apply the boundary condition to the
            if(fvLabel || subDomain.InRegion(cells[0])){
                PetscScalar *xI;
                PetscScalar *xG;

                DMPlexPointLocalRead(dm, cells[0], xArray, &xI) >> checkError;
                DMPlexPointLocalFieldRef(dm, cells[1], fieldId.id, xArray, &xG) >> checkError;
                updateFunction(time, fg->centroid, fg->normal, xI, xG, updateContext);
            }else{
                // flip the ghost cells and normal (updateFunction assumes normal points to ghost cell)
                PetscScalar *xI;
                PetscScalar *xG;
                DMPlexPointLocalRead(dm, cells[1], xArray, &xI) >> checkError;
                DMPlexPointLocalFieldRef(dm, cells[0], fieldId.id, xArray, &xG) >> checkError;

                PetscReal   flipedNormal[3];
                for(PetscInt d =0; d < dim; d++){
                    flipedNormal[d] = -fg->normal[d];
                }
                updateFunction(time, fg->centroid, flipedNormal, xI, xG, updateContext);
            }
        }

        ISRestoreIndices(faceIS, &faces) >> checkError;
        ISDestroy(&faceIS) >> checkError;
    }

    VecRestoreArray(locXVec, &xArray) >> checkError;
    VecRestoreArrayRead(cellGeometryVec, &cellGeometryArray) >> checkError;
    VecRestoreArrayRead(faceGeometryVec, &faceGeometryArray) >> checkError;
}

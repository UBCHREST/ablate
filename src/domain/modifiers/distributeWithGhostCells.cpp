#include "distributeWithGhostCells.hpp"
#include <utilities/petscError.hpp>

ablate::domain::modifiers::DistributeWithGhostCells::DistributeWithGhostCells(int ghostCellDepthIn) : ghostCellDepth(ghostCellDepthIn < 1 ? 2 : ghostCellDepthIn) {}
void ablate::domain::modifiers::DistributeWithGhostCells::Modify(DM &dm) {
    // Make sure that the flow is set up distributed
    DM dmDist;
    // create any ghost cells that are needed
    DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE) >> checkError;
    DMPlexDistribute(dm, ghostCellDepth, NULL, &dmDist) >> checkError;
    ReplaceDm(dm, dmDist);
    TagMpiGhostCells(dm) >> checkError;
}

PetscErrorCode ablate::domain::modifiers::DistributeWithGhostCells::TagMpiGhostCells(DM dmNew) {
    PetscSF sfPoint;
    DMLabel ghostLabel = NULL;
    const PetscSFNode *leafRemote;
    const PetscInt *leafLocal;
    PetscInt cellHeight, cStart, cEnd, c, fStart, fEnd, f, numLeaves, l;
    PetscMPIInt rank;

    PetscFunctionBegin;
    /* Step 11: Make label for output (vtk) and to mark ghost points (ghost) */
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dmNew), &rank));
    PetscCall(DMGetPointSF(dmNew, &sfPoint));
    PetscCall(DMPlexGetVTKCellHeight(dmNew, &cellHeight));
    PetscCall(DMPlexGetHeightStratum(dmNew, cellHeight, &cStart, &cEnd));
    PetscCall(PetscSFGetGraph(sfPoint, NULL, &numLeaves, &leafLocal, &leafRemote));
    PetscCall(DMCreateLabel(dmNew, "ghost"));
    PetscCall(DMGetLabel(dmNew, "ghost", &ghostLabel));

    DMLabel vtkLabel;
    DMLabelCreate(PETSC_COMM_SELF, "vtkLabel", &vtkLabel) >> checkError;

    for (l = 0, c = cStart; l < numLeaves && c < cEnd; ++l, ++c) {
        for (; c < leafLocal[l] && c < cEnd; ++c) {
            PetscCall(DMLabelSetValue(vtkLabel, c, 1));
        }
        if (leafLocal[l] >= cEnd) break;
        if (leafRemote[l].rank == rank) {
            PetscCall(DMLabelSetValue(vtkLabel, c, 1));
        } else if (ghostLabel) {
            PetscCall(DMLabelSetValue(ghostLabel, c, 2));
        }
    }
    for (; c < cEnd; ++c) {
        PetscCall(DMLabelSetValue(vtkLabel, c, 1));
    }
    if (ghostLabel) {
        PetscCall(DMPlexGetHeightStratum(dmNew, 1, &fStart, &fEnd));
        for (f = fStart; f < fEnd; ++f) {
            PetscInt numCells;

            PetscCall(DMPlexGetSupportSize(dmNew, f, &numCells));
            if(numCells == 1){
                const PetscInt *cells = NULL;
                PetscInt vA;
                PetscCall(DMPlexGetSupport(dmNew, f, &cells));
                PetscCall(DMLabelGetValue(vtkLabel, cells[0], &vA));
                if(vA!= 1) PetscCall(DMLabelSetValue(ghostLabel, f, 1));
            }
            if (numCells  > 1) {
                const PetscInt *cells = NULL;
                PetscInt vA, vB;

                PetscCall(DMPlexGetSupport(dmNew, f, &cells));
                PetscCall(DMLabelGetValue(vtkLabel, cells[0], &vA));
                PetscCall(DMLabelGetValue(vtkLabel, cells[1], &vB));
                if (vA != 1 && vB != 1) PetscCall(DMLabelSetValue(ghostLabel, f, 1));
            }
        }
    }

    DMLabelDestroy(&vtkLabel) >> checkError;
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::DistributeWithGhostCells, "Distribute DMPlex with ghost cells",
         OPT(int, "ghostCellDepth", "the number of ghost cells to share on the boundary.  Default is 1."));
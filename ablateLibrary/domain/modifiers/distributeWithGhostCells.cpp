#include "distributeWithGhostCells.hpp"
#include <utilities/petscError.hpp>

ablate::domain::modifiers::DistributeWithGhostCells::DistributeWithGhostCells(int ghostCellDepthIn) : ghostCellDepth(ghostCellDepthIn < 1 ? 1 : ghostCellDepthIn) {}
void ablate::domain::modifiers::DistributeWithGhostCells::Modify(DM &dm) {
    // Make sure that the flow is setup distributed
    DM dmDist;
    // create any ghost cells that are needed
    DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE) >> checkError;
    DMPlexDistribute(dm, ghostCellDepth, NULL, &dmDist) >> checkError;
    if (dmDist) {
        DMDestroy(&dm) >> checkError;
        dm = dmDist;
    }
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::DistributeWithGhostCells, "Distribute DMPlex with ghost cells",
         OPT(int, "ghostCellDepth", "the number of ghost cells to share on the boundary.  Default is 1."));
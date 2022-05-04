#include "distributeWithGhostCells.hpp"
#include <utilities/petscError.hpp>

ablate::domain::modifiers::DistributeWithGhostCells::DistributeWithGhostCells(int ghostCellDepthIn) : ghostCellDepth(ghostCellDepthIn < 1 ? 2 : ghostCellDepthIn) {}
void ablate::domain::modifiers::DistributeWithGhostCells::Modify(DM &dm) {
    // Make sure that the flow is setup distributed
    DM dmDist;
    // create any ghost cells that are needed
    DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE) >> checkError;
    DMPlexDistribute(dm, ghostCellDepth, NULL, &dmDist) >> checkError;
    if (dmDist) {
        // Copy over the options object
        PetscOptions options;
        PetscObjectGetOptions((PetscObject)dm, &options) >> checkError;
        PetscObjectSetOptions((PetscObject)dmDist, options) >> checkError;

        DMDestroy(&dm) >> checkError;
        dm = dmDist;
    }
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::DistributeWithGhostCells, "Distribute DMPlex with ghost cells",
         OPT(int, "ghostCellDepth", "the number of ghost cells to share on the boundary.  Default is 1."));
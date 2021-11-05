#include "ghostBoundaryCells.hpp"
#include <utilities/petscError.hpp>
ablate::domain::modifiers::GhostBoundaryCells::GhostBoundaryCells(std::string labelName) : labelName(labelName) {}
void ablate::domain::modifiers::GhostBoundaryCells::Modify(DM &dm) {
    DM gdm;
    DMPlexConstructGhostCells(dm, labelName.empty() ? nullptr : labelName.c_str(), NULL, &gdm) >> checkError;
    DMDestroy(&dm) >> checkError;
    dm = gdm;
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::GhostBoundaryCells, "Adds ghost cells to the boundary",
         OPT(std::string, "labelName", "The label specifying the boundary faces, or \"Face Sets\" if not specified"));
#include "zeroGradientGhost.hpp"
ablate::flow::boundaryConditions::ZeroGradientGhost::ZeroGradientGhost(std::string fieldName, std::string boundaryName, std::vector<int> labelIds, std::string labelName)
    : Ghost(fieldName, boundaryName, labelIds, ZeroGradientGhostUpdate, this, labelName) {}

PetscErrorCode ablate::flow::boundaryConditions::ZeroGradientGhost::ZeroGradientGhostUpdate(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG,
                                                                                            void *ctx) {
    PetscFunctionBeginUser;
    ablate::flow::boundaryConditions::ZeroGradientGhost *ghost = (ablate::flow::boundaryConditions::ZeroGradientGhost *)ctx;

    for (PetscInt i = 0; i < ghost->fieldSize; i++) {
        a_xG[i] = a_xI[ghost->fieldOffset + i];
    }
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::boundaryConditions::BoundaryCondition, ablate::flow::boundaryConditions::ZeroGradientGhost, "fills the ghost cells with the flow cell resulting in zero normal gradient",
         ARG(std::string, "fieldName", "the name of the field"), ARG(std::string, "boundaryName", "the name for this boundary condition"),
         ARG(std::vector<int>, "labelIds", "the ids on the mesh to apply the boundary condition"), OPT(std::string, "labelName", "the mesh label holding the boundary ids (default Face Sets)"));
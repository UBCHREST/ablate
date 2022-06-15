#include "essentialGhost.hpp"
ablate::finiteVolume::boundaryConditions::EssentialGhost::EssentialGhost(std::string boundaryName, std::vector<int> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction,
                                                                         std::string labelName, bool enforceAtFace)
    : Ghost(boundaryFunction->GetName(), boundaryName, labelIds, EssentialGhostUpdate, this, labelName), boundaryFunction(boundaryFunction), enforceAtFace(enforceAtFace) {}
PetscErrorCode ablate::finiteVolume::boundaryConditions::EssentialGhost::EssentialGhostUpdate(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG,
                                                                                              void *ctx) {
    PetscFunctionBeginUser;
    // cast the pointer back to a math function
    ablate::finiteVolume::boundaryConditions::EssentialGhost *essentialGhost = (ablate::finiteVolume::boundaryConditions::EssentialGhost *)ctx;

    // Use the petsc function directly
    PetscCall(essentialGhost->boundaryFunction->GetSolutionField().GetPetscFunction()(
        essentialGhost->dim, time, c, essentialGhost->fieldSize, a_xG, essentialGhost->boundaryFunction->GetSolutionField().GetContext()));

    if (essentialGhost->enforceAtFace) {
        // use linear interpolation to enforce at face
        for (PetscInt f = 0; f < essentialGhost->fieldSize; f++) {
            a_xG[f] *= 2.0;
            a_xG[f] -= a_xI[essentialGhost->fieldOffset + f];
        }
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::boundaryConditions::BoundaryCondition, ablate::finiteVolume::boundaryConditions::EssentialGhost, "essential (Dirichlet condition) for ghost cell based boundaries",
         ARG(std::string, "boundaryName", "the name for this boundary condition"), ARG(std::vector<int>, "labelIds", "the ids on the mesh to apply the boundary condition"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary"),
         OPT(std::string, "labelName", "the mesh label holding the boundary ids (default Face Sets)"),
         OPT(bool, "enforceAtFace", "optionally update the boundary to enforce the value at the face instead of the cell using linear interpolation (default falue)"));
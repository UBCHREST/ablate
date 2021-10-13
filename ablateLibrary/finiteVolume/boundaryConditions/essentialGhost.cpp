#include "essentialGhost.hpp"
ablate::flow::boundaryConditions::EssentialGhost::EssentialGhost(std::string boundaryName, std::vector<int> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction,
                                                                 std::string labelName)
    : Ghost(boundaryFunction->GetName(), boundaryName, labelIds, EssentialGhostUpdate, this, labelName), boundaryFunction(boundaryFunction) {}
PetscErrorCode ablate::flow::boundaryConditions::EssentialGhost::EssentialGhostUpdate(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    // cast the pointer back to a math function
    ablate::flow::boundaryConditions::EssentialGhost *essentialGhost = (ablate::flow::boundaryConditions::EssentialGhost *)ctx;

    // Use the petsc function directly
    // PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx
    return essentialGhost->boundaryFunction->GetSolutionField().GetPetscFunction()(
        essentialGhost->dim, time, c, essentialGhost->fieldSize, a_xG, essentialGhost->boundaryFunction->GetSolutionField().GetContext());
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::boundaryConditions::BoundaryCondition, ablate::flow::boundaryConditions::EssentialGhost, "essential (Dirichlet condition) for ghost cell based boundaries",
         ARG(std::string, "boundaryName", "the name for this boundary condition"), ARG(std::vector<int>, "labelIds", "the ids on the mesh to apply the boundary condition"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary"),
         OPT(std::string, "labelName", "the mesh label holding the boundary ids (default Face Sets)"));
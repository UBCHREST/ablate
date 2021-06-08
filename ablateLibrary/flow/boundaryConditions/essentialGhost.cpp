#include "essentialGhost.hpp"
ablate::flow::boundaryConditions::EssentialGhost::EssentialGhost(std::string fieldName, std::string boundaryName, std::string labelName, std::vector<int> labelIds,
                                                                 std::shared_ptr<mathFunctions::MathFunction> boundaryFunction)
    : Ghost(fieldName, boundaryName, labelName, labelIds, EssentialGhostUpdate, this), boundaryFunction(boundaryFunction) {}
PetscErrorCode ablate::flow::boundaryConditions::EssentialGhost::EssentialGhostUpdate(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    // cast the pointer back to a math function
    ablate::flow::boundaryConditions::EssentialGhost*  essentialGhost = (ablate::flow::boundaryConditions::EssentialGhost*)ctx;

    // Use the petsc function directly
    // PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx
    return essentialGhost->boundaryFunction->GetPetscFunction()(essentialGhost->dim, time, c, essentialGhost->fieldSize, a_xG, essentialGhost->boundaryFunction->GetContext());
}

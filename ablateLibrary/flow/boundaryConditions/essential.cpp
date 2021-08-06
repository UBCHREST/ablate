#include "essential.hpp"
#include <utilities/petscError.hpp>

PetscErrorCode ablate::flow::boundaryConditions::Essential::BoundaryValueFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    auto boundary = (Essential *)ctx;
    return boundary->boundaryFunction->GetSolutionField().GetPetscFunction()(dim, time, x, Nf, u, boundary->boundaryFunction->GetSolutionField().GetContext());
}
PetscErrorCode ablate::flow::boundaryConditions::Essential::BoundaryTimeDerivativeFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    auto boundary = (Essential *)ctx;
    return boundary->boundaryFunction->GetTimeDerivative().GetPetscFunction()(dim, time, x, Nf, u, boundary->boundaryFunction->GetTimeDerivative().GetContext());
}

ablate::flow::boundaryConditions::Essential::Essential(std::string boundaryName, int labelId, std::shared_ptr<mathFunctions::FieldFunction> boundaryFunctionIn, std::string labelNameIn)
    : Essential(boundaryName, std::vector<int>{labelId}, boundaryFunctionIn, labelNameIn) {}

ablate::flow::boundaryConditions::Essential::Essential(std::string boundaryName, std::vector<int> labelId, std::shared_ptr<mathFunctions::FieldFunction> boundaryFunctionIn, std::string labelNameIn)
    : BoundaryCondition(boundaryName, boundaryFunctionIn->GetName()), labelName(labelNameIn.empty() ? "marker" : labelNameIn), labelIds(labelId), boundaryFunction(boundaryFunctionIn) {}

ablate::mathFunctions::PetscFunction ablate::flow::boundaryConditions::Essential::GetBoundaryFunction() { return BoundaryValueFunction; }
ablate::mathFunctions::PetscFunction ablate::flow::boundaryConditions::Essential::GetBoundaryTimeDerivativeFunction() { return BoundaryTimeDerivativeFunction; }

void *ablate::flow::boundaryConditions::Essential::GetContext() { return this; }

void ablate::flow::boundaryConditions::Essential::SetupBoundary(PetscDS problem, PetscInt fieldId) {
    PetscDSAddBoundary(problem,
                       DM_BC_ESSENTIAL,
                       GetBoundaryName().c_str(),
                       GetLabelName().c_str(),
                       fieldId,
                       0,
                       NULL,
                       (void (*)(void))GetBoundaryFunction(),
                       (void (*)(void))GetBoundaryTimeDerivativeFunction(),
                       labelIds.size(),
                       &labelIds[0],
                       GetContext()) >>
        checkError;
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::boundaryConditions::BoundaryCondition, ablate::flow::boundaryConditions::Essential, "essential (Dirichlet condition) for FE based problems",
         ARG(std::string, "boundaryName", "the name for this boundary condition"), ARG(std::vector<int>, "labelIds", "the ids on the mesh to apply the boundary condition"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue", "the field function used to describe the boundary"),
         OPT(std::string, "labelName", "the mesh label holding the boundary ids (default marker)"));
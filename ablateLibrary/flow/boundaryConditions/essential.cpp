#include "essential.hpp"
#include <utilities/petscError.hpp>

PetscErrorCode ablate::flow::boundaryConditions::Essential::BoundaryValueFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    auto boundary = (Essential *)ctx;
    return boundary->boundaryValue->GetPetscFunction()(dim, time, x, Nf, u, boundary->boundaryValue->GetContext());
}
PetscErrorCode ablate::flow::boundaryConditions::Essential::BoundaryTimeDerivativeFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    auto boundary = (Essential *)ctx;
    return boundary->timeDerivativeValue->GetPetscFunction()(dim, time, x, Nf, u, boundary->timeDerivativeValue->GetContext());
}

ablate::flow::boundaryConditions::Essential::Essential(std::string fieldName, std::string boundaryName, std::string labelName, int labelId, std::shared_ptr<mathFunctions::MathFunction> boundaryValue,
                                                       std::shared_ptr<mathFunctions::MathFunction> timeDerivativeValue)
    : BoundaryCondition(boundaryName, fieldName), labelName(labelName), labelIds({labelId}), boundaryValue(boundaryValue), timeDerivativeValue(timeDerivativeValue) {}

ablate::flow::boundaryConditions::Essential::Essential(std::string fieldName, std::string boundaryName, std::string labelName, std::vector<int> labelId,
                                                       std::shared_ptr<mathFunctions::MathFunction> boundaryValue, std::shared_ptr<mathFunctions::MathFunction> timeDerivativeValue)
    : BoundaryCondition(boundaryName, fieldName), labelName(labelName), labelIds(labelId), boundaryValue(boundaryValue), timeDerivativeValue(timeDerivativeValue) {}

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
         ARG(std::string, "fieldName", "the field to apply the boundary condition"),
         ARG(std::string, "boundaryName", "the name for this boundary condition"),
         ARG(std::string, "labelName", "the mesh label holding the boundary ids"),
         ARG(std::vector<int>, "labelIds", "the ids on the mesh to apply the boundary condition"),
         ARG(ablate::mathFunctions::MathFunction, "boundaryValue", "the math function used to describe the boundary"),
         OPT(ablate::mathFunctions::MathFunction, "timeDerivativeValue", "the math function used to describe the field time derivative"));
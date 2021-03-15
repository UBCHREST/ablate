#include "boundaryCondition.hpp"
#include "parser/registrar.hpp"

PetscErrorCode ablate::flow::BoundaryCondition::BoundaryValueFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    auto boundary = (BoundaryCondition *)ctx;
    return boundary->boundaryValue->GetPetscFunction()(dim, time, x, Nf, u, boundary->boundaryValue->GetContext());
}
PetscErrorCode ablate::flow::BoundaryCondition::BoundaryTimeDerivativeFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    auto boundary = (BoundaryCondition *)ctx;
    return boundary->timeDerivativeValue->GetPetscFunction()(dim, time, x, Nf, u, boundary->timeDerivativeValue->GetContext());
}

ablate::mathFunctions::PetscFunction ablate::flow::BoundaryCondition::GetBoundaryFunction() { return BoundaryValueFunction; }
ablate::mathFunctions::PetscFunction ablate::flow::BoundaryCondition::GetBoundaryTimeDerivativeFunction() { return BoundaryTimeDerivativeFunction; }

void *ablate::flow::BoundaryCondition::GetContext() { return this; }

ablate::flow::BoundaryCondition::BoundaryCondition(std::string fieldName, std::string boundaryName, std::string labelName, int labelId, std::shared_ptr<mathFunctions::MathFunction> boundaryValue,
                                                   std::shared_ptr<mathFunctions::MathFunction> timeDerivativeValue)
    : fieldName(fieldName), boundaryName(boundaryName), labelName(labelName), labelId(labelId), bcType(DM_BC_ESSENTIAL), boundaryValue(boundaryValue), timeDerivativeValue(timeDerivativeValue) {}

REGISTERDEFAULT(ablate::flow::BoundaryCondition, ablate::flow::BoundaryCondition, "a description of the flow field boundary condition", ARG(std::string, "fieldName", "the field name"),
                ARG(std::string, "boundaryName", "the name of this boundary condition"), ARG(std::string, "labelName", "the label to used to determine boundary nodes"),
                ARG(int, "labelId", "the label id"), ARG(ablate::mathFunctions::MathFunction, "boundaryValue", "the math function used to describe the boundary"),
                ARG(ablate::mathFunctions::MathFunction, "timeDerivativeValue", "the math function used to describe the field time derivative"));

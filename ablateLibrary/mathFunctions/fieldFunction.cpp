#include "fieldFunction.hpp"
#include "registrar.hpp"

ablate::mathFunctions::FieldFunction::FieldFunction(std::string fieldName, std::shared_ptr<mathFunctions::MathFunction> solutionField, std::shared_ptr<mathFunctions::MathFunction> timeDerivative)
    : solutionField(solutionField), timeDerivative(timeDerivative), fieldName(fieldName) {}

REGISTER_DEFAULT(ablate::mathFunctions::FieldFunction, ablate::mathFunctions::FieldFunction, "a field description that can be used for initialization or exact solution ",
                 ARG(std::string, "fieldName", "the field name"), ARG(ablate::mathFunctions::MathFunction, "field", "the math function used to describe the field"),
                 OPT(ablate::mathFunctions::MathFunction, "timeDerivative", "the math function used to describe the field time derivative"));

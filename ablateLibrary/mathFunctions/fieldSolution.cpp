#include "fieldSolution.hpp"
#include "parser/registrar.hpp"

ablate::mathFunctions::FieldSolution::FieldSolution(std::string fieldName, std::shared_ptr<mathFunctions::MathFunction> solutionField, std::shared_ptr<mathFunctions::MathFunction> timeDerivative)
    : solutionField(solutionField), timeDerivative(timeDerivative), fieldName(fieldName) {}

REGISTERDEFAULT(ablate::mathFunctions::FieldSolution, ablate::mathFunctions::FieldSolution, "a field description that can be used for initialization or exact solution ",
                ARG(std::string, "fieldName", "the field name"), ARG(mathFunctions::MathFunction, "solutionField", "the math function used to describe the field"),
                OPT(mathFunctions::MathFunction, "timeDerivative", "the math function used to describe the field time derivative"));

#include "flowFieldSolution.hpp"
#include "parser/registrar.hpp"

ablate::flow::FlowFieldSolution::FlowFieldSolution(std::string fieldName, std::shared_ptr<mathFunctions::MathFunction> solutionField, std::shared_ptr<mathFunctions::MathFunction> timeDerivative)
    : fieldName(fieldName), solutionField(solutionField), timeDerivative(timeDerivative) {}

REGISTERDEFAULT(ablate::flow::FlowFieldSolution, ablate::flow::FlowFieldSolution, "a field description that can be used for initialization or exact solution ",
                ARG(std::string, "fieldName", "the field name"), ARG(mathFunctions::MathFunction, "solutionField", "the math function used to describe the field"),
                OPT(mathFunctions::MathFunction, "timeDerivative", "the math function used to describe the field time derivative"));

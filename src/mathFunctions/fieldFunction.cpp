#include "fieldFunction.hpp"

#include <utility>

ablate::mathFunctions::FieldFunction::FieldFunction(std::string fieldName, std::shared_ptr<mathFunctions::MathFunction> solutionField, std::shared_ptr<mathFunctions::MathFunction> timeDerivative,
                                                    std::shared_ptr<ablate::domain::Region> region)
    : solutionField(std::move(solutionField)), timeDerivative(std::move(timeDerivative)), fieldName(std::move(fieldName)), region(std::move(region)) {}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::mathFunctions::FieldFunction, ablate::mathFunctions::FieldFunction, "a field description that can be used for initialization or exact solution ",
                 ARG(std::string, "fieldName", "the field name"),
                 ARG(ablate::mathFunctions::MathFunction, "field", "the math function used to describe the field"),
                 OPT(ablate::mathFunctions::MathFunction, "timeDerivative", "the math function used to describe the field time derivative"),
                 OPT(ablate::domain::Region, "region", "A subset of the domain to apply the field function"));

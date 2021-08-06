#ifndef ABLATELIBRARY_FIELDFUNCTION_HPP
#define ABLATELIBRARY_FIELDFUNCTION_HPP
#include <memory>
#include <string>
#include "mathFunctions/mathFunction.hpp"

namespace ablate::mathFunctions {
class FieldFunction {
   private:
    const std::shared_ptr<mathFunctions::MathFunction> solutionField;
    const std::shared_ptr<mathFunctions::MathFunction> timeDerivative;
    const std::string fieldName;

   public:
    FieldFunction(std::string fieldName, std::shared_ptr<mathFunctions::MathFunction> solutionField, std::shared_ptr<mathFunctions::MathFunction> timeDerivative = {});
    virtual ~FieldFunction() = default;

    const std::string& GetName() const { return fieldName; }

    bool HasSolutionField() const { return solutionField != nullptr; }
    mathFunctions::MathFunction& GetSolutionField() { return *solutionField; }

    bool HasTimeDerivative() const { return timeDerivative != nullptr; }
    mathFunctions::MathFunction& GetTimeDerivative() { return *timeDerivative; }

    std::shared_ptr<mathFunctions::MathFunction> GetFieldFunction() const { return solutionField; }
    std::shared_ptr<mathFunctions::MathFunction> GetTimeDerivativeFunction() const { return timeDerivative; }
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_FIELDFUNCTION_HPP

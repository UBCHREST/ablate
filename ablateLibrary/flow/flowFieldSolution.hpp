#ifndef ABLATELIBRARY_FLOWFIELDSOLUTION_HPP
#define ABLATELIBRARY_FLOWFIELDSOLUTION_HPP
#include <memory>
#include <string>
#include "mathFunctions/mathFunction.hpp"

namespace ablate::flow {
class FlowFieldSolution {
   private:
    const std::shared_ptr<mathFunctions::MathFunction> solutionField;
    const std::shared_ptr<mathFunctions::MathFunction> timeDerivative;
    const std::string fieldName;

   public:
    FlowFieldSolution(std::string fieldName, std::shared_ptr<mathFunctions::MathFunction> solutionField, std::shared_ptr<mathFunctions::MathFunction> timeDerivative);

    const std::string& GetName() const { return fieldName; }

    mathFunctions::MathFunction& GetSolutionField() { return *solutionField; }

    mathFunctions::MathFunction& GetTimeDerivative() { return *timeDerivative; }
};
}  // namespace ablate::flow
#endif  // ABLATELIBRARY_FLOWFIELDSOLUTION_HPP

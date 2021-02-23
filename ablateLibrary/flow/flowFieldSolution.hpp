#ifndef ABLATELIBRARY_FLOWFIELDSOLUTION_HPP
#define ABLATELIBRARY_FLOWFIELDSOLUTION_HPP
#include <memory>
#include "mathFunctions/mathFunction.hpp"
#include <string>

namespace ablate::flow {
class FlowFieldSolution {
   private:
    const std::shared_ptr<mathFunctions::MathFunction> solutionField;
    const std::shared_ptr<mathFunctions::MathFunction> timeDerivative;
    const std::string fieldName;

   public:
    FlowFieldSolution(std::string fieldName, std::shared_ptr<mathFunctions::MathFunction> solutionField, std::shared_ptr<mathFunctions::MathFunction> timeDerivative);

    const std::string& GetName() const{
        return fieldName;
    }

    mathFunctions::MathFunction& GetSolutionField() {
        return *solutionField;
    }

    mathFunctions::MathFunction& GetTimeDerivative(){
        return *timeDerivative;
    }
};
}
#endif  // ABLATELIBRARY_FLOWFIELDSOLUTION_HPP

#ifndef ABLATELIBRARY_ESSENTIAL_HPP
#define ABLATELIBRARY_ESSENTIAL_HPP

#include <memory>
#include <string>
#include "mathFunctions/mathFunction.hpp"
#include "boundaryCondition.hpp"

namespace ablate::flow::boundaryConditions {

class Essential: public BoundaryCondition {
   private:
    const std::string labelName;
    const std::vector<int> labelIds;
    const std::shared_ptr<mathFunctions::MathFunction> boundaryValue;
    const std::shared_ptr<mathFunctions::MathFunction> timeDerivativeValue;

   private:
    static PetscErrorCode BoundaryValueFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);
    static PetscErrorCode BoundaryTimeDerivativeFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

    mathFunctions::PetscFunction GetBoundaryFunction();

    mathFunctions::PetscFunction GetBoundaryTimeDerivativeFunction();

    void* GetContext();

   public:
    Essential(std::string fieldName, std::string boundaryName, std::string labelName, int labelId, std::shared_ptr<mathFunctions::MathFunction> boundaryValue,
    std::shared_ptr<mathFunctions::MathFunction> timeDerivativeValue = nullptr);

    Essential(std::string fieldName, std::string boundaryName, std::string labelName, std::vector<int> labelId, std::shared_ptr<mathFunctions::MathFunction> boundaryValue,
    std::shared_ptr<mathFunctions::MathFunction> timeDerivativeValue = nullptr);

    const std::string& GetLabelName() const { return labelName; }

    const std::vector<int>& GetLabelIds() const { return labelIds; }

    void SetupBoundary(PetscDS problem, PetscInt field) override;
};

}
#endif  // ABLATELIBRARY_ESSENTIAL_HPP

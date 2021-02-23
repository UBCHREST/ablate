#ifndef ABLATELIBRARY_BOUNDARYCONDITION_HPP
#define ABLATELIBRARY_BOUNDARYCONDITION_HPP
#include <memory>
#include "mathFunctions/mathFunction.hpp"
#include <string>
namespace ablate::flow {
class BoundaryCondition {
   private:
    const std::string fieldName;
    const std::string boundaryName;
    const std::string labelName;
    const int labelId;
    const std::shared_ptr<mathFunctions::MathFunction> boundaryValue;
    const std::shared_ptr<mathFunctions::MathFunction> timeDerivativeValue;
    const int bcType;

   private:
    static PetscErrorCode BoundaryValueFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
    static PetscErrorCode BoundaryTimeDerivativeFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);

   public:
    BoundaryCondition(std::string fieldName, std::string boundaryName, std::string labelName, int labelId, std::shared_ptr<mathFunctions::MathFunction> boundaryValue, std::shared_ptr<mathFunctions::MathFunction> timeDerivativeValue);

    const std::string& GetFieldName() const{
        return fieldName;
    }

    const std::string& GetBoundaryName() const{
        return boundaryName;
    }

    const std::string& GetLabelName() const{
        return labelName;
    }

    const int& GetLabelId() const{
        return labelId;
    }

    const int& GetBCType() const{
        return bcType;
    }

    mathFunctions::PetscFunction GetBoundaryFunction();

    mathFunctions::PetscFunction GetBoundaryTimeDerivativeFunction();

    void* GetContext();
};
}
#endif  // ABLATELIBRARY_BOUNDARYCONDITION_HPP

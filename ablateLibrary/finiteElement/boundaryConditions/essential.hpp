#ifndef ABLATELIBRARY_ESSENTIAL_HPP
#define ABLATELIBRARY_ESSENTIAL_HPP

#include <mathFunctions/fieldFunction.hpp>
#include <memory>
#include <string>
#include "boundaryCondition.hpp"
#include "mathFunctions/mathFunction.hpp"

namespace ablate::finiteElement::boundaryConditions {

class Essential : public BoundaryCondition {
   private:
    const std::string labelName;
    const std::vector<PetscInt> labelIds;
    const std::shared_ptr<mathFunctions::FieldFunction> boundaryFunction;

   private:
    static PetscErrorCode BoundaryValueFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);
    static PetscErrorCode BoundaryTimeDerivativeFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

    mathFunctions::PetscFunction GetBoundaryFunction();

    mathFunctions::PetscFunction GetBoundaryTimeDerivativeFunction();

    void* GetContext();

   public:
    Essential(std::string boundaryName, int labelId, std::shared_ptr<mathFunctions::FieldFunction> boundaryFunction, std::string labelName = {});

    Essential(std::string boundaryName, std::vector<int> labelId, std::shared_ptr<mathFunctions::FieldFunction> boundaryFunction, std::string labelName = {});

    const std::string& GetLabelName() const { return labelName; }

    void SetupBoundary(DM dm, PetscDS problem, PetscInt field) override;
};

}  // namespace ablate::flow::boundaryConditions
#endif  // ABLATELIBRARY_ESSENTIAL_HPP

#ifndef ABLATELIBRARY_BOUNDARYCONDITION_HPP
#define ABLATELIBRARY_BOUNDARYCONDITION_HPP
#include <memory>
#include <string>
#include "flow/flowFieldDescriptor.hpp"
#include "mathFunctions/mathFunction.hpp"

namespace ablate::flow::boundaryConditions {
class BoundaryCondition {
   private:
    const std::string boundaryName;
    const std::string fieldName;

   protected:
    BoundaryCondition(const std::string boundaryName, const std::string fieldName) : boundaryName(boundaryName), fieldName(fieldName) {}

   public:
    const std::string& GetBoundaryName() const { return boundaryName; }
    const std::string& GetFieldName() const { return fieldName; }

    virtual ~BoundaryCondition() = default;
    virtual void SetupBoundary(DM dm, PetscDS problem, PetscInt fieldId) = 0;
};
}  // namespace ablate::flow::boundaryConditions
#endif  // ABLATELIBRARY_BOUNDARYCONDITION_HPP

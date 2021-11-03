#ifndef ABLATELIBRARY_FINITEVOLUME_BOUNDARYCONDITION_HPP
#define ABLATELIBRARY_FINITEVOLUME_BOUNDARYCONDITION_HPP
#include <memory>
#include <string>
#include "domain/fieldDescriptor.hpp"
#include "mathFunctions/mathFunction.hpp"

namespace ablate::finiteVolume::boundaryConditions {
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
}  // namespace ablate::finiteVolume::boundaryConditions
#endif  // ABLATELIBRARY_BOUNDARYCONDITION_HPP

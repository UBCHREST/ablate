#ifndef ABLATELIBRARY_GHOST_HPP
#define ABLATELIBRARY_GHOST_HPP

#include "boundaryCondition.hpp"

namespace ablate::finiteVolume::boundaryConditions {

class Ghost : public BoundaryCondition {
    typedef PetscErrorCode (*UpdateFunction)(PetscReal time, const PetscReal* c, const PetscReal* n, const PetscScalar* a_xI, PetscScalar* a_xG, void* ctx);

   private:
    const std::string labelName;
    const std::vector<PetscInt> labelIds;
    const UpdateFunction updateFunction;
    const void* updateContext;

   protected:
    // Store some field information
    PetscInt dim;
    PetscInt fieldSize;
    // the field offset for the a_xI values;
    PetscInt fieldOffset;

   public:
    Ghost(std::string fieldName, std::string boundaryName, std::vector<int> labelIds, UpdateFunction updateFunction, void* updateContext, std::string labelName = {});

    Ghost(std::string fieldName, std::string boundaryName, int labelId, UpdateFunction updateFunction, void* updateContext, std::string labelName = {});

    virtual ~Ghost() override = default;

    void SetupBoundary(DM dm, PetscDS problem, PetscInt fieldId) override;
};

}  // namespace ablate::flow::boundaryConditions

#endif  // ABLATELIBRARY_GHOST_HPP

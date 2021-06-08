#ifndef ABLATELIBRARY_GHOST_HPP
#define ABLATELIBRARY_GHOST_HPP

#include "boundaryCondition.hpp"

namespace ablate::flow::boundaryConditions {

class Ghost : public BoundaryCondition {
    typedef PetscErrorCode (*UpdateFunction)(PetscReal time, const PetscReal* c, const PetscReal* n, const PetscScalar* a_xI, PetscScalar* a_xG, void* ctx);

   private:
    const std::string labelName;
    const std::vector<int> labelIds;
    const UpdateFunction updateFunction;
    const void* updateContext;

   protected:
    // Store some field information
    PetscInt dim;
    PetscInt fieldSize;

   public:
    Ghost(std::string fieldName, std::string boundaryName, std::string labelName, std::vector<int> labelIds, UpdateFunction updateFunction, void* updateContext);

    Ghost(std::string fieldName, std::string boundaryName, std::string labelName, int labelId, UpdateFunction updateFunction, void* updateContext);

    virtual ~Ghost() override = default;

    void SetupBoundary(PetscDS problem, PetscInt field) override;
};

}  // namespace ablate::flow::boundaryConditions

#endif  // ABLATELIBRARY_GHOST_HPP

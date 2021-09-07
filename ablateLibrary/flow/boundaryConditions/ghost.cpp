#include "ghost.hpp"
#include <petsc.h>
#include <utilities/petscError.hpp>

ablate::flow::boundaryConditions::Ghost::Ghost(std::string fieldName, std::string boundaryName, std::vector<int> labelIds, ablate::flow::boundaryConditions::Ghost::UpdateFunction updateFunction,
                                               void *updateContext, std::string labelNameIn)
    : BoundaryCondition(boundaryName, fieldName),
      labelName(labelNameIn.empty() ? "Face Sets" : labelNameIn),
      labelIds(labelIds.begin(), labelIds.end()),
      updateFunction(updateFunction),
      updateContext(updateContext) {}

ablate::flow::boundaryConditions::Ghost::Ghost(std::string fieldName, std::string boundaryName, int labelId, ablate::flow::boundaryConditions::Ghost::UpdateFunction updateFunction,
                                               void *updateContext, std::string labelName)
    : Ghost(fieldName, boundaryName, std::vector<int>{labelId}, updateFunction, updateContext, labelName) {}

void ablate::flow::boundaryConditions::Ghost::SetupBoundary(PetscDS problem, PetscInt fieldId) {
    PetscDSAddBoundary(
        problem, DM_BC_NATURAL_RIEMANN, GetBoundaryName().c_str(), labelName.c_str(), fieldId, 0, NULL, (void (*)(void))updateFunction, NULL, labelIds.size(), &labelIds[0], (void *)updateContext) >>
        checkError;

    // extract some information about the flowField
    PetscDSGetFieldSize(problem, fieldId, &fieldSize) >> checkError;
    PetscDSGetCoordinateDimension(problem, &dim) >> checkError;
    PetscDSGetFieldOffset(problem, fieldId, &fieldOffset) >> checkError;
}

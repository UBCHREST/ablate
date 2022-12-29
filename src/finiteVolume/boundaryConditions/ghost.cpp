#include "ghost.hpp"
#include <petsc.h>
#include "utilities/petscUtilities.hpp"

ablate::finiteVolume::boundaryConditions::Ghost::Ghost(std::string fieldName, std::string boundaryName, std::vector<int> labelIds,
                                                       ablate::finiteVolume::boundaryConditions::Ghost::UpdateFunction updateFunction, void *updateContext, std::string labelNameIn)
    : BoundaryCondition(boundaryName, fieldName),
      labelName(labelNameIn.empty() ? "Face Sets" : labelNameIn),
      labelIds(labelIds.begin(), labelIds.end()),
      updateFunction(updateFunction),
      updateContext(updateContext) {}

ablate::finiteVolume::boundaryConditions::Ghost::Ghost(std::string fieldName, std::string boundaryName, int labelId, ablate::finiteVolume::boundaryConditions::Ghost::UpdateFunction updateFunction,
                                                       void *updateContext, std::string labelName)
    : Ghost(fieldName, boundaryName, std::vector<int>{labelId}, updateFunction, updateContext, labelName) {}

void ablate::finiteVolume::boundaryConditions::Ghost::SetupBoundary(DM dm, PetscDS problem, PetscInt fieldId) {
    DMLabel label;
    DMGetLabel(dm, labelName.c_str(), &label) >> utilities::PetscUtilities::checkError;
    PetscDSAddBoundary(
        problem, DM_BC_NATURAL_RIEMANN, GetBoundaryName().c_str(), label, labelIds.size(), &labelIds[0], fieldId, 0, NULL, (void (*)(void))updateFunction, NULL, (void *)updateContext, NULL) >>
        utilities::PetscUtilities::checkError;

    // extract some information about the flowField
    PetscDSGetFieldSize(problem, fieldId, &fieldSize) >> utilities::PetscUtilities::checkError;
    PetscDSGetCoordinateDimension(problem, &dim) >> utilities::PetscUtilities::checkError;
    PetscDSGetFieldOffset(problem, fieldId, &fieldOffset) >> utilities::PetscUtilities::checkError;
}

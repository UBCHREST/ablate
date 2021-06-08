#include "ghost.hpp"
#include <utilities/petscError.hpp>
#include <petsc.h>

ablate::flow::boundaryConditions::Ghost::Ghost(std::string fieldName, std::string boundaryName, std::string labelName, std::vector<int> labelIds,
                                               ablate::flow::boundaryConditions::Ghost::UpdateFunction updateFunction, void *updateContext) : BoundaryCondition(boundaryName, fieldName),labelName(labelName), labelIds(labelIds), updateFunction(updateFunction), updateContext(updateContext) {}

void ablate::flow::boundaryConditions::Ghost::SetupBoundary(PetscDS problem, PetscInt fieldId) {
    PetscDSAddBoundary(problem,
                       DM_BC_NATURAL_RIEMANN,
                       GetBoundaryName().c_str(),
                       labelName.c_str(),
                       fieldId,
                       0,
                       NULL,
                       (void (*)(void))updateFunction,
                       NULL,
                       labelIds.size(),
                       &labelIds[0],
                       (void*)updateContext) >>
                                     checkError;

    // extract some information about the flowField
    PetscDSGetFieldSize(problem, fieldId, &fieldSize) >> checkError;
    PetscDSGetCoordinateDimension(problem, &dim) >> checkError;
}
